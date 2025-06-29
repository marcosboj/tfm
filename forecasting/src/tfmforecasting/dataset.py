from enum import StrEnum


class HousingUnitColumns(StrEnum):
    Date = "date"
    Time = "time"
    Consumption = "consumptionKWh"


class AdditionalHousingUnitFields(StrEnum):
    Datetime = "datetime"
    HousingUnit = "housing_unit"
