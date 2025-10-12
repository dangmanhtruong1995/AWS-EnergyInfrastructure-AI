from dataclasses import dataclass, field
from typing import Set, List

from pydantic import BaseModel


@dataclass
class DataSourceTracker:
    used_sources: Set[str] = field(default_factory=set)
    report: Set[str] = field(default_factory=set)


class ReportMapOutput(BaseModel):
    """ Output with report and map"""
    report: str
    map: str

class AvailableDataSources(BaseModel):
    """ A list of available data sources."""
    data_source_list: List[str]


class GetWellEntryInput(BaseModel):
    """Input for getting the report of well entry"""
    location: str


class WellEntryOutput(BaseModel):
    """Output of the well entry, given as a report."""
    report: str


class DataSourceOutput(BaseModel):
    """ Input for getting the data source (seismic, well, etc.) """
    source_name: str


class SeismicAndDrillingInput(BaseModel):
    """Input for plotting the seismic and drilling data"""
    seismic_data_path: str
    well_data_path: str


class SeismicAndDrillingOutput(BaseModel):
    """Output for plotting the seismic and drilling data. final_report: The report (string)."""
    final_report: str
    html_content: str


class SeismicAndLicensedBlocksInput(BaseModel):
    """Input for analyzing and plotting the seismic and licensed blocks data"""
    seismic_data_path: str
    license_block_data_path: str


class AnalysisOutput(BaseModel):
    """ Output of the analysis function, which is a report in string format. """
    status: str


class PlotOutput(BaseModel):
    """ Output of the plot function. status: 'Success' or 'Failed' """
    status: str