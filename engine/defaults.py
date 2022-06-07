"""
Boilerplate logic for controlling training and evaluation processes.
Author: JiaWei Jiang
"""
import argparse
from abc import abstractmethod
from argparse import Namespace
from typing import Optional

__all__ = ["TrainEvalArgParser", "InferArgParser"]


class BaseArgParser:
    """Base class for all customized argument parsers.

    Todo - Move to base directory.
    """

    def __init__(self) -> None:
        self.argparser = argparse.ArgumentParser()
        self._build()

    def parse(self) -> Namespace:
        """Return arguments driving the designated processes.

        Return:
            args: arguments driving the designated processes
        """
        args = self.argparser.parse_args()
        return args

    @abstractmethod
    def _build(self) -> None:
        """Build argument parser."""
        raise NotImplementedError

    def _str2bool(self, arg: str) -> Optional[bool]:
        """Convert boolean argument from string representation into
        bool.

        Parameters:
            arg: str, argument in string representation

        Return:
            True or False: bool, argument in bool dtype
        """
        # See https://stackoverflow.com/questions/15008758/
        # parsing-boolean-values-with-argparse
        if isinstance(arg, bool):
            return arg
        if arg.lower() in ("true", "t", "yes", "y", "1"):
            return True
        elif arg.lower() in ("false", "f", "no", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError(
                "Expect boolean representation " "for argument --eval-only."
            )


class TrainEvalArgParser(BaseArgParser):
    """Argument parser for training and evaluation processes."""

    def __init__(self) -> None:
        super().__init__()

    def _build(self) -> None:
        """Build argument parser."""
        self.argparser.add_argument(
            "--project-name",
            type=str,
            default=None,
            help="name of the project configuring wandb project",
        )
        self.argparser.add_argument(
            "--input-path",
            type=str,
            default=None,
            help="path of the input file",
        )
        self.argparser.add_argument(
            "--model-name",
            type=str,
            default=None,
            help="name of the model architecture",
        )
        self.argparser.add_argument(
            "--cv-scheme",
            type=str,
            default=None,
            help="cross-validation scheme",
        )
        self.argparser.add_argument(
            "--n-folds", type=int, default=None, help="total number of folds"
        )
        self.argparser.add_argument(
            "--oof-size",
            type=int,
            default=None,
            help="numeber of dates in oof",
        )
        self.argparser.add_argument(
            "--max-train-size",
            type=int,
            default=None,
            help="max numeber of dates in training set",
        )
        self.argparser.add_argument(
            "--group", type=str, default=None, help="column to group CV folds"
        )
        self.argparser.add_argument(
            "--stratified",
            type=str,
            default=None,
            help="column acting as stratified determinant",
        )
        self.argparser.add_argument(
            "--train-ratio",
            type=float,
            default=None,
            help="ratio of training samples",
        )
        self.argparser.add_argument(
            "--val-ratio",
            type=float,
            default=None,
            help="ratio of validation samples",
        )
        self.argparser.add_argument(
            "--random-state",
            type=int,
            default=None,
            help="random state seeding shuffling process of cross validator",
        )


class InferArgParser(BaseArgParser):
    """Argument parser for inference process."""

    def __init__(self) -> None:
        super().__init__()

    def _build(self) -> None:
        """Build argument parser."""
        self.argparser.add_argument(
            "--project-name",
            type=str,
            default=None,
            help="name of the project configuring wandb project",
        )
        self.argparser.add_argument(
            "--exp-id",
            type=str,
            default=None,
            help="experiment identifier used to group experiment entries",
        )
        self.argparser.add_argument(
            "--input-path",
            type=str,
            default=None,
            help="path of the input file",
        )
        self.argparser.add_argument(
            "--model-name",
            type=str,
            default=None,
            help="name of the model architecture",
        )
        self.argparser.add_argument(
            "--model-version",
            type=str,
            default=None,
            help="version of the model used to predict",
        )
        self.argparser.add_argument(
            "--model-type",
            type=str,
            default=None,
            choices=["fold", "whole"],
            help="type of the models used to predict",
        )
        self.argparser.add_argument(
            "--model-ids",
            type=int,
            default=None,
            nargs="*",
            help="identifiers of models used to predict (fold numbers or "
            "seed numbers)",
        )
        self.argparser.add_argument(
            "--device",
            type=str,
            default="cuda:0",
            help="device used to run the designated processes",
        )
