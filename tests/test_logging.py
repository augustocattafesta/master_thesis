import datetime

from loguru import logger

from analysis import ANALYSIS_RESULTS


def test():
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.add(ANALYSIS_RESULTS / date, level="INFO")

    logger.info("Hi file")

test()