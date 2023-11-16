import json
import logging
import os
import re
import sys

from utils.file_utils import encode_image, load_txt_file
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

        self.prompt = self.config["initial_prompt"]
        self.log(f"initializing prompt to `{self.prompt}`")

    def pre_processing(self):
        # load problem images into encoded images
        solution_files = os.listdir(self.solutions_path)
        problem_statement_files = os.listdir(self.problem_statements_path)

        # verify solution and problem filenames are correct
        assert len(solution_files) == len(problem_statement_files), "difference between number of solutions"
        for fn in solution_files:
            assert re.match(r"^solution[0-9]+\..*", fn), f"invalid solution file name {fn}"
        for fn in problem_statement_files:
            assert re.match(r"^problem[0-9]+\..*", fn), f"invalid problem file name {fn}"

        # sort by problem number in filename
        solution_files.sort(
            key=lambda fn: int(fn[8:].split(".")[0])
        )
        problem_statement_files.sort(
            key=lambda fn: int(fn[7:].split(".")[0])
        )

        self.problem_statements = []
        self.solutions = []
        for problem_statement_fn, solution_fn in zip(problem_statement_files, solution_files):
            # Construct the full file path
            problem_statement_path = os.path.join(self.problem_statements_path, problem_statement_fn)
            solution_path = os.path.join(self.solutions_path, solution_fn)

            # TODO: make more general to handle
            self.problem_statements.append(encode_image(problem_statement_path))
            self.solutions.append(load_txt_file(solution_path))

        self.log(f"loaded {len(self.solutions)} training problem(s)")

    def do_training(self):
        pass

    def get_logfilepath(self):
        """
        get the log path (under the software path)
        Returns:
            start_time: time for starting making the software
            log_filepath: path to the log

        """
        start_time = now()
        filepath = os.path.dirname(__file__)
        root = os.path.dirname(filepath)
        directory = os.path.join(root, "Workspace")
        directory = os.path.join(directory, self.solution_version)
        directory = os.path.join(directory, "logs")
        os.makedirs(directory, exist_ok=True)
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
