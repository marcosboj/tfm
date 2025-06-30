from enum import StrEnum
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeAlias
from urllib.parse import urlencode

import aiofiles
from aiohttp import ClientSession

BASE_URL = "https://mandeo.meteogalicia.es"
NET_CDF_SUBSET = "/thredds/ncss/grid/modelos/WRF_HIST"

ThreddsWRFDomain: TypeAlias = Literal["d02"]
ThreddsWRFServerRun: TypeAlias = Literal["0000", "1200"]


class MeteoGaliciaWRFVar(StrEnum):
    PRECIPITATION = "prec"
    TEMP = "temp"
    SWFLX = "swflx"


class MeteoGaliciaNetCDFSubsetColumns(StrEnum):
    DATE = "date"
    TEMP = 'temp[unit="K"]'
    CFL = 'cfl[unit="1"]'
    CFM = 'cfm[unit="1"]'
    MOD = 'mod[unit="m s-1"]'
    SWFLX = 'swflx[unit="W m-2"]'


class NetCDFSubsetParameterBuilder:
    date_format = "%Y-%m-%d"

    def __init__(self, variables: list[MeteoGaliciaWRFVar]) -> None:
        self._parameters: dict[str, str | list] = {"var": variables}

    @property
    def query_dict(self) -> dict[str, str]:
        d = {}
        for key, value in self._parameters.items():
            if type(value) is list:
                d[key] = ",".join(value)
            else:
                d[key] = str(value)
        return d

    def add_date_range(
            self, start: datetime, end: datetime
    ):
        self._parameters["time_start"] = start.strftime(self.date_format)
        self._parameters["time_end"] = end.strftime(self.date_format)
        return self

    def add_point(self, point: tuple[float, float]):
        self._parameters["point"] = 'true'
        self._parameters["latitude"] = str(point[0])
        self._parameters["longitude"] = str(point[1])
        return self

    def accept_csv(self):
        self._parameters["accept"] = "csv"
        return self


def get_wrf_arw_det_history_filename(domain: ThreddsWRFDomain, server_run: ThreddsWRFServerRun, date: datetime):
    date_format = "%Y%m%d"
    return f"wrf_arw_det_history_{domain}_{date.strftime(date_format)}_{server_run}"


def get_net_cdf_subset_filename(domain: ThreddsWRFDomain, server_run: ThreddsWRFServerRun, date: datetime) -> str:
    return f"{get_wrf_arw_det_history_filename(domain, server_run, date)}.nc4"


def net_cdf_subset_url(
        day: datetime,
        domain: ThreddsWRFDomain,
        server_run: ThreddsWRFServerRun,
        query_parameters: dict[str, str]
) -> str:
    query_str = urlencode(query_parameters)
    net_cdf_subset_file = get_net_cdf_subset_filename(domain, server_run, day)
    wrf_file = f"/{domain}/{day.year}/{day.month:02}/{net_cdf_subset_file}"
    return f"{BASE_URL}{NET_CDF_SUBSET}{wrf_file}?{query_str}"


# async functions
async def fetch_net_cdf_subset(url: str, session: ClientSession) -> str:
    async with session.get(url) as response:
        return await response.text()


async def write_contents_to_file(file: Path, url: str, session: ClientSession) -> None:
    content = await fetch_net_cdf_subset(url=url, session=session)
    if not content:
        return
    async with aiofiles.open(file, mode="w") as f:
        await f.write(content)
