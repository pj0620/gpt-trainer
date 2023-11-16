import json
import logging
import os
import sys

from utils.time_utils import now


class LearningModel:

    def __init__(self,
                 problem_config_path: str = None,
                 problem_statements_path: str = None,
                 solutions_path: str = None,
                 problem_name: str = None,
                 solution_version: str = "v1.0.0",
                 model_type: str = "gpt-4-vision-preview") -> None:
        """

        Args:
            problem_config_path: path to the problem_config.json
            problem_statements_path: path to input problem statement images
            solutions_path: path to solutions
            problem_name: name of problem being solved
            model_type: gpt model type
        """

        # load config file
        self.problem_config_path = problem_config_path
        self.problem_statements_path = problem_statements_path
        self.solutions_path = solutions_path
        self.problem_name = problem_name
        self.solution_version = solution_version
        self.model_type = model_type

        with open(self.problem_config_path, 'r', encoding="utf8") as file:
            self.config = json.load(file)

        self.start_time, self.log_filepath = self.get_logfilepath()

        self.logger = self.build_logger()

    def get_logfilepath(self):
        """
        get the log path (under the software path)
        Returns:
            start_time: time for starting making the software
            log_filepath: path to the log

        """
        start_time = now()
        filepath = os.path.dirname(__file__)
        # root = "/".join(filepath.split("/")[:-1])
        root = os.path.dirname(filepath)
        # directory = root + "/WareHouse/"
        directory = os.path.join(root, "Workspace")
        log_filepath = os.path.join(directory,
                                    "{}.log".format("_".join([self.problem_name, self.solution_version, start_time])))
        return start_time, log_filepath

    def build_logger(self):
        # Create a logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Create a file handler which logs messages to a file
        file_handler = logging.FileHandler(self.log_filepath, encoding="utf-8")
        file_formatter = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s', datefmt='%Y-%d-%m %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Create a stream handler to log messages to stdout
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_formatter = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s', datefmt='%Y-%d-%m %H:%M:%S')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        return logger

    def log(self, msg: str):
        self.logger.info(msg)
