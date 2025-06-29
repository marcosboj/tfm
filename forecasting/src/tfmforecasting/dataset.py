from enum import StrEnum


class HouseholdColumns(StrEnum):
    Date = "date"
    Time = "time"
    Consumption = "consumptionKWh"


class AdditionalHouseholdFields(StrEnum):
    Datetime = "datetime"
