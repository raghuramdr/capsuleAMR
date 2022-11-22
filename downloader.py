import argparse
import asyncio
import dataclasses
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import aiofiles
import httpx
import pandas as pd
from httpx import AsyncClient

from colored_logger import LoggerContextManager
from async_bgzf import AsyncBgzfWriter, AsyncBgzfReader

OLD_GID_COLUMN_NAME = "Genome ID"
OLD_ANTIBIOTIC_COLUMN_NAME = "Antibiotic"
GID_COLUMN_NAME = "genome_id"
ANTIBIOTIC_COLUMN_NAME = "antibiotic"

GENOME_FOLDER_NAME = "genomes"


@dataclasses.dataclass
class GenomeInformation:
    genome_id: str
    antibiotic: str


def read_file(csv_path: Path) -> pd.DataFrame:
    """
    Read the csv file containing the genome IDs.
    If the csv file contains a column named "Genome ID",
    rename it "genome_id".

    Parameters
    ----------
    csv_path : pathlib.Path
        Full path to the CSV file containing the genome IDs

    Returns
    -------
    df : pd.DataFrame
        Pandas dataframe containing genome IDs
    """

    if not csv_path.exists() or not csv_path.is_file():
        logger.critical(f"File {csv_path} does not exist")
        exit(1)
    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip", dtype={OLD_GID_COLUMN_NAME: str},
                         usecols=[OLD_GID_COLUMN_NAME, OLD_ANTIBIOTIC_COLUMN_NAME])
    except ValueError as e:
        logger.info(
            f"The columns '{OLD_GID_COLUMN_NAME}' and '{OLD_ANTIBIOTIC_COLUMN_NAME}' are missing in the csv file.")
        raise SystemExit(e)
    logger.info("Successfully read the data file.")

    df.rename(columns={OLD_GID_COLUMN_NAME: GID_COLUMN_NAME, OLD_ANTIBIOTIC_COLUMN_NAME: ANTIBIOTIC_COLUMN_NAME},
              inplace=True)
    df[ANTIBIOTIC_COLUMN_NAME] = df[ANTIBIOTIC_COLUMN_NAME].str.replace("/", "_")
    df[ANTIBIOTIC_COLUMN_NAME] = df[ANTIBIOTIC_COLUMN_NAME].str.replace(" ", "-")
    return df


def compute_stats(df: pd.DataFrame):
    """
    Print some stats about the genomes

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with all genome and antibiotic combinations
    """
    logger.info(f"{len(df)} of genome id and antibiotic combinations")
    download_count = df[GID_COLUMN_NAME].nunique()
    logger.info(f"{download_count} ({100 * download_count / len(df):.2f}%) of unique genome ids")
    logger.info(f"{df[ANTIBIOTIC_COLUMN_NAME].nunique()} of unique antibiotic")


def set_parameters() -> argparse.Namespace:
    """
    Function to accept user defined parameters.

    Returns
    ------
    config : argparse.Namespace
        Config parse from the commandline
    """

    parser = argparse.ArgumentParser(
        description="Read CSV files and download FASTA files corresponding to the genome IDs in the CSV files.")
    parser.add_argument("--log-path", type=Path, required=False, dest="log_path",
                        help="Path to directory where to save the logs. Default is the current working directory.")
    parser.add_argument("--data-path", type=Path, required=False, dest="data_path",
                        help="Path where to create the antibiotic folders. If None the folder of the csv file is used.")
    parser.add_argument("--csv-file", type=Path, required=True, help="Path to the CSV file.", dest="csv_file")
    parser.add_argument("--redownload", action="store_true", help="Force to re-download the genomes.")
    parser.add_argument("--max-connections", type=int, required=False, default=20, dest="max_connections",
                        help="Number of TCP Connections used for downloading. Default 20.")
    parser.add_argument("--sleep-time", type=int, required=False, default=3, dest="sleep_time",
                        help="Time in seconds to sleep before downloading a genome. Default 3.")
    parser.add_argument("--max-retry", type=int, required=False, default=5, dest="max_retry",
                        help="Number of times the script retries to download a genome after an error. Default 5.")
    parser.add_argument("--no-compress", action="store_false", dest="compress",
                        help="Force to save raw file instead of compressing it")
    args = parser.parse_args()
    return args


def preprocess_genome_ids(df: pd.DataFrame) -> Tuple[List[str], List[GenomeInformation]]:
    """
    Select the files which should be downloaded and which symlinks to create

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing all genome_id and antibiotic combinations

    Returns
    -------
    genome_download_info_list : list[GenomeInformation]
        All combinations of genome_id and antibiotic for which the genome should be downloaded
    genome_symlink_info_list : list[GenomeInformation]
        All combinations of genome_id and antibiotic for which the genome can be linked
    """
    genome_download_ids = list(df[GID_COLUMN_NAME].unique())
    genome_symlink_info_list: List[GenomeInformation] = []
    genome_antibiotic_combination_set = set()
    for _, row in df.iterrows():
        genome_antibiotic_hash = hash(row[GID_COLUMN_NAME] + row[ANTIBIOTIC_COLUMN_NAME])
        # If we already encountered the genome_id / antibiotic combination, ignore this one
        if genome_antibiotic_hash in genome_antibiotic_combination_set:
            continue
        genome_antibiotic_combination_set.add(genome_antibiotic_hash)

        # If we already encountered the genome_id, reference it
        genome_symlink_info_list.append(GenomeInformation(genome_id=row[GID_COLUMN_NAME], antibiotic=row[ANTIBIOTIC_COLUMN_NAME]))
    return list(genome_download_ids), genome_symlink_info_list


async def download_file(asyncio_semaphore: asyncio.BoundedSemaphore, client: AsyncClient,
                        genome_id: str, genomes_path: Path, logger: logging.Logger,
                        config: argparse.Namespace):
    """
    Download a genome

    Parameters
    ----------
    asyncio_semaphore : asyncio.BoundedSemaphore
        BoundedSemaphore to ensure that only a number of Coroutine a active at the same time
    client : httpx.AsyncClient
        Client to download the genome
    genome_id : str
        ID of genome to download.
    genomes_path : pathlib.Path
        Directory path where to save the genome
    logger : logging.Logger
        Logger for logging information
    config : argparse.Namespace
        Config with additional information
    """
    file_path = genomes_path.joinpath(GENOME_FOLDER_NAME,
                                      f"{genome_id}.fa{'.bz2' if config.compress else ''}")
    complement_compression_file_path = genomes_path.joinpath(GENOME_FOLDER_NAME,
                                                             f"{genome_id}.fa{'' if config.compress else '.bz2'}")
    # if the file exist we don't download it unless we force it
    if config.redownload:
        pass
    elif file_path.exists():
        logger.info(f"Genome {genome_id} in {file_path} already exist. Skip...")
        return
    elif complement_compression_file_path.exists():
        # (Un-)Compressed file already exists, so we don't need to download the genome and just (de-)compress the file
        async with asyncio_semaphore:
            logger.info(f"File {complement_compression_file_path} exists. Use this file instead of downloading")
            if complement_compression_file_path.name.endswith("bz2"):
                # file is compressed
                async with aiofiles.open(complement_compression_file_path, mode='rb') as fin, aiofiles.open(file_path, mode='wb') as fout:
                    bgzf_reader = AsyncBgzfReader(fin)
                    await bgzf_reader.start()
                    async for line in bgzf_reader:
                        await fout.write(line)
                    await bgzf_reader.close()
            else:
                # file is uncompressed
                async with aiofiles.open(complement_compression_file_path, mode='rb') as fin, aiofiles.open(file_path, mode='wb') as fout:
                    bgzf_writer = AsyncBgzfWriter(fout)
                    chunk = await fin.read(65536)
                    while chunk:
                        await bgzf_writer.write(chunk)
                        chunk = await fin.read(65536)
                    await bgzf_writer.close()
        return

    # Download the genome
    retry_counter = 0
    data = {
        "rql": f"in(genome_id%2C({genome_id}))%26sort(%2Bsequence_id)%26limit(2500000)",
    }
    while retry_counter < config.max_retry:
        try:
            async with asyncio_semaphore:
                await asyncio.sleep(float(config.sleep_time))  # Sleep to avoid to many request at once
                if retry_counter > 0:
                    logger.info(
                        f"Retry download genome {genome_id} to {file_path} {retry_counter + 1}/{config.max_retry}")
                else:
                    logger.info(f"Download genome {genome_id} to {file_path}")
                async with client.stream("POST",
                                         url="https://patricbrc.org/api/genome_sequence/?&http_download=true&http_accept=application/dna+fasta",
                                         data=data) as resp, aiofiles.open(file_path, mode='wb') as f:
                    async_bgzf_writer = AsyncBgzfWriter(async_fileobj=f)
                    async for chunk in resp.aiter_bytes():
                        if config.compress:
                            await async_bgzf_writer.write(data=chunk)
                        else:
                            await f.write(chunk)
                    if config.compress:
                        await async_bgzf_writer.close()
                logger.info(f"FASTA file for genome ID {genome_id} successfully written to {file_path}")
                break
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as _:
            retry_counter += 1
            logger.error(
                f"Error when downloading genome {genome_id}. {'Retry to download.' if retry_counter < config.max_retry else 'Skip this genome. Try again later.'}")
            file_path.unlink(missing_ok=True)


async def download_genomes(genome_id_list: List[str], genomes_path: Path, logger: logging.Logger,
                           config: argparse.Namespace):
    """
    Wrapper function for managing the async download of the genomes.

    Parameters
    ----------
    genome_id_list : list[str]
        IDs of all the  genomes to download.
    genomes_path : pathlib.Path
        Directory path where to save the genomes
    logger : logging.Logger
        Logger for logging information
    config : argparse.Namespace
        Config with additional information
    """
    headers = {
        'Connection': 'keep-alive',
    }
    # Limit the TCP Connections
    tcp_limits = httpx.Limits(max_connections=config.max_connections, max_keepalive_connections=20)
    client = AsyncClient(headers=headers, limits=tcp_limits, timeout=10)
    # Limit the number of Coroutine that a running at the same time to the number of TCP Connection to avoid a Timeout
    asyncio_semaphore = asyncio.BoundedSemaphore(config.max_connections)
    tasks = []
    for genome_id in genome_id_list:
        tasks.append(
            asyncio.ensure_future(download_file(asyncio_semaphore, client, genome_id, genomes_path, logger, config)))

    logger.info("Start downloading genomes")
    await asyncio.gather(*tasks)


def create_symlinks(genome_list: List[GenomeInformation], genomes_path: Path, logger: logging.Logger,
                    config: argparse.Namespace):
    """
    Create Symlinks to downloaded genomes.

    Parameters
    ----------
    genome_list : list[GenomeInformation]
        Information about all the  genomes to download.
    genomes_path : pathlib.Path
        Directory path where to save the genomes
    logger : logging.Logger
        Logger for logging information
    config : argparse.Namespace
        Config with additional information
    """
    for genome_info in genome_list:
        file_path = genomes_path.joinpath(genome_info.antibiotic, f"{genome_info.genome_id}.fa{'.bz2' if config.compress else ''}")
        if file_path.exists():
            logger.info(f"Genome {genome_info.genome_id} in {file_path} already exist. Skip...")
            continue
        link_file = genomes_path.joinpath(GENOME_FOLDER_NAME, f"{genome_info.genome_id}.fa{'.bz2' if config.compress else ''}")
        file_path.symlink_to(link_file.absolute())
        logger.info(f"{file_path.name} already downloaded. Link to {link_file}")


if __name__ == "__main__":
    config = set_parameters()
    with LoggerContextManager(config.log_path) as logger:
        logger.info(f"The filename is {config.csv_file}")
        df = read_file(config.csv_file)
        df.dropna(subset=["genome_id"], inplace=True)
        data_path = config.data_path if config.data_path else config.csv_file.parent

        compute_stats(df)

        for antibiotic in df[ANTIBIOTIC_COLUMN_NAME].unique():
            antibiotic_path = data_path.joinpath(antibiotic)
            logger.info(f"Create folder {antibiotic_path}")
            antibiotic_path.mkdir(exist_ok=True)
        data_path.joinpath(GENOME_FOLDER_NAME).mkdir(exist_ok=True)

        genomes_download, genomes_symlink = preprocess_genome_ids(df)

        asyncio.run(download_genomes(genomes_download, data_path, logger, config))

        create_symlinks(genomes_symlink, data_path, logger, config)

        logger.info("Program execution finished!")
