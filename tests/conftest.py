import pytest

from rockfall.config import Config, load_config
from rockfall.pipeline import Pipeline
from rockfall.site import Site, demo_site


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config()


@pytest.fixture(scope="session")
def site() -> Site:
    return demo_site()


@pytest.fixture(scope="session")
def pipeline(cfg: Config, site: Site) -> Pipeline:
    return Pipeline(cfg, site)
