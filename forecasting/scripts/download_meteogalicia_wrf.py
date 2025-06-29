import asyncio
from datetime import timezone
from pathlib import Path

import pandas as pd
from aiohttp import TCPConnector, ClientSession

from tfmmeteogalicia.thredds_wrf import (
    MeteoGaliciaWRFVar, NetCDFSubsetParameterBuilder, get_wrf_arw_det_history_filename, net_cdf_subset_url,
    write_contents_to_file
)


async def main() -> None:
    parallel_connections = 4
    data_dir = Path("../data/meteogalicia/thredds/wrf_hist")
    dates = [
        pd.Timestamp(year=2024, month=1, day=18, hour=0, minute=0, second=0, tzinfo=timezone.utc),
        pd.Timestamp(year=2024, month=1, day=19, hour=0, minute=0, second=0, tzinfo=timezone.utc),
    ]
    latitude, longitude = 43.20342940827081, -2.0500748424398574
    weather_vars = [MeteoGaliciaWRFVar.TEMP]
    conn = TCPConnector(limit=parallel_connections)
    async with ClientSession(connector=conn) as session:
        for date in dates:
            parameters = NetCDFSubsetParameterBuilder(
                weather_vars
            ).add_date_range(
                start=date,
                end=date + pd.Timedelta(days=4),
            ).add_point(
                (latitude, longitude)
            ).accept_csv(
            ).query_dict
            url = net_cdf_subset_url(date, domain="d02", server_run="0000", query_parameters=parameters)
            print(url)
            await write_contents_to_file(
                data_dir / f"{get_wrf_arw_det_history_filename(domain="d02", server_run="0000", date=date)}.csv", url,
                session
            )


if __name__ == "__main__":
    asyncio.run(main())
