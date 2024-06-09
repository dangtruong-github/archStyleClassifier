import configparser

from pipeline_all.naive_classification import (
    NaiveClassificationPipeline
)

config = configparser.ConfigParser()

config.read('config.ini')

NaiveClassificationPipeline(config)
