from datetime import datetime
from pathlib import Path

import pandas as pd

from tfmmeteogalicia.thredds_wrf import ThreddsWRFDomain, ThreddsWRFServerRun, get_wrf_arw_det_history_filename, \
    MeteoGaliciaNetCDFSubsetColumns


def load_wrf_hist_dataset(
        data_dir: Path, domain: ThreddsWRFDomain, server_run: ThreddsWRFServerRun, date: datetime
) -> pd.DataFrame:
    csv_path = data_dir / f'{get_wrf_arw_det_history_filename(domain, server_run, date)}.csv'
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df[MeteoGaliciaNetCDFSubsetColumns.DATE] = pd.to_datetime(df[MeteoGaliciaNetCDFSubsetColumns.DATE])
    return df
