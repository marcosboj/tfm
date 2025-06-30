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
    dates = pd.date_range(start="2024-01-01", end="2024-06-01", freq="D", tz=timezone.utc)
    latitude, longitude = 41.6524241, -0.9280126
    weather_vars = [MeteoGaliciaWRFVar.TEMP, MeteoGaliciaWRFVar.PRECIPITATION, MeteoGaliciaWRFVar.SWFLX]
    conn = TCPConnector(limit=parallel_connections)
    async with ClientSession(connector=conn) as session:
        for date in dates:
            parameters = NetCDFSubsetParameterBuilder(
                weather_vars
            ).add_date_range(
                start=date,
                end=date + pd.Timedelta(days=2),
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
