from typing import Tuple, Dict, Optional, Any, List
import numpy as np
import os
import h5py
from datetime import datetime
from copy import deepcopy
import json

from tndm.utils import logger, CustomEncoder


class DataManager(object):

    def apply_split(data: np.ndarray,
                    indices: List[np.ndarray]) -> List[np.ndarray]:
        """Array Split

        Splits the provided array on the first axis according to the indices provided.

        Args:
            data (np.ndarray): Input array
            indices (List[np.ndarray]): List of indices

        Returns:
            List[np.ndarray]: List of output arrays
        """
        splits = [data[tuple([index] + [slice(None)] * (data.ndim - 1))]
                  for index in indices]
        return splits

    @staticmethod
    def split_dataset(
            data: np.ndarray, train_pct: float = None, valid_pct: float = None, test_pct: float = None) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray]:
        """Split Dataset

        Splits a numpy array into train, validation and test based on the first axis.

        Args:
            data (np.ndarray): input dataset
            train_pct (float, optional): percentage of records used for training. Defaults to None.
            valid_pct (float, optional): percentage of records used for validation. Defaults to None.
            test_pct (float, optional): percentage of records used for testing. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: training dataset,
                validation dataset, test dataset, training index, validation index and test index.
        """

        tot_pct = sum(
            [x for x in [train_pct, valid_pct, test_pct] if x is not None])
        if tot_pct <= 1:
            missing = sum(
                [x is None for x in [train_pct, valid_pct, test_pct]])
            if missing > 1:
                raise ValueError(
                    'Cannot handle more than one missing percentage.')
            elif missing == 1:
                [train_pct, valid_pct, test_pct] = [x if x is not None else 1 -
                                                    tot_pct for x in [train_pct, valid_pct, test_pct]]
            else:
                train_pct /= tot_pct
                valid_pct /= tot_pct
                test_pct /= tot_pct
        else:
            missing = sum(
                [x is None for x in [train_pct, valid_pct, test_pct]])
            if missing > 1:
                raise ValueError(
                    'Cannot handle more than one missing percentage.')
            elif missing == 1:
                [train_pct, valid_pct, test_pct] = [
                    x / 100 if x is not None else 1 - tot_pct / 100 for x in [train_pct, valid_pct, test_pct]]
            else:
                train_pct /= tot_pct
                valid_pct /= tot_pct
                test_pct /= tot_pct

        index = np.arange(data.shape[0])
        np.random.shuffle(index)
        train_i, validation_i, test_i = np.split(index,
                                                 [int(train_pct * data.shape[0]), int((valid_pct + train_pct) * data.shape[0])])

        train, validation, test = DataManager.apply_split(
            data, [train_i, validation_i, test_i])

        return (train, validation, test, train_i, validation_i, test_i)

    @staticmethod
    def build_dataset(neural_data: np.ndarray, behaviour_data: np.ndarray, settings: Dict[str, Any],
                      noisless_behaviour_data: Optional[np.ndarray] = None, rates_data: Optional[np.ndarray] = None,
                      latent_data: Optional[np.ndarray] = None, time_data: Optional[np.ndarray] = None,
                      behaviour_weights: Optional[np.ndarray] = None, neural_weights: Optional[np.ndarray] = None,
                      train_pct: float = None, valid_pct: float = None, test_pct: float = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Builds the dataset

        Dimensions:
            K: trials
            T: timesteps
            N: neurons
            Y: behavioural dimension
            B: behaviour relevant latent variables
            L: neural latent variables (behaviour relevant + behaviour irrelevant)

        Args:
            neural_data (np.ndarray): matrix of spikes (size: K x T x N)
            behaviour_data (np.ndarray): matrix of behaviour (size: K x T x Y)
            settings (Dict[str, Any]): settings dictionary
            noisless_behaviour_data (Optional[np.ndarray], optional): matrix of behaviour before adding Gaussian
                noise (size: K x T x Y). Defaults to None.
            rates_data (Optional[np.ndarray], optional): matrix of firing rates (size: K x T x N).
                Defaults to None.
            latent_data (Optional[np.ndarray], optional): matrix of latent trajectories (size: K x T x 3).
                Defaults to None.
            time_data (Optional[np.ndarray], optional): time vector (size: T). Defaults to None.
            behaviour_weights (Optional[np.ndarray], optional): weights to convert latent variables
                into noisless behaviour (size: B x Y). Defaults to None.
            neural_weights (Optional[np.ndarray], optional): weights to convert latent variables into firing
                rates (size: L x N). Defaults to None.
            train_pct (float, optional): percentage of training data. Defaults to None.
            valid_pct (float, optional): percentage of validation data. Defaults to None.
            test_pct (float, optional): percentage of test data. Defaults to None.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Dataset as key-value pair and settings dict.
        """
        ######## STANDARD PROPERTIES ########
        data_dict: Dict[str, Any] = {}

        data_dict['train_data'], data_dict['valid_data'], data_dict['test_data'], train_i, valid_i, test_i = \
            DataManager.split_dataset(
                neural_data, train_pct=train_pct, valid_pct=valid_pct, test_pct=test_pct)

        data_dict['train_behaviours'], data_dict['valid_behaviours'], data_dict['test_behaviours'] = \
            DataManager.apply_split(behaviour_data, indices=[
                                    train_i, valid_i, test_i])

        settings_out = deepcopy(settings)
        settings_out.update({
            'train_pct': train_pct,
            'valid_pct': valid_pct,
            'test_pct': test_pct,
            'created': datetime.utcnow().isoformat()})

        ######## OPTIONAL PROPERTIES ########
        if noisless_behaviour_data is not None:
            data_dict['train_behaviours_noiseless'], data_dict['valid_behaviours_noiseless'], data_dict['test_behaviours_noiseless'] = \
                DataManager.apply_split(noisless_behaviour_data, indices=[
                                        train_i, valid_i, test_i])

        if rates_data is not None:
            data_dict['train_rates'], data_dict['valid_rates'], data_dict['test_rates'] = \
                DataManager.apply_split(rates_data, indices=[
                                        train_i, valid_i, test_i])

        if latent_data is not None:
            data_dict['train_latent'], data_dict['valid_latent'], data_dict['test_latent'] = \
                DataManager.apply_split(latent_data, indices=[
                                        train_i, valid_i, test_i])

        if time_data is not None:
            data_dict['time_data'] = time_data

        if behaviour_weights is not None:
            data_dict['relevant_dims'] = np.ndarray(behaviour_weights.shape[0])
            data_dict['behaviour_weights'] = behaviour_weights

        if neural_weights is not None:
            data_dict['neural_weights'] = neural_weights

        return data_dict, settings_out

    @staticmethod
    def store_dataset(dataset: Dict[str, np.ndarray], settings: Dict[str, Any],
                      directory: str, filename: Optional[str] = 'dataset.h5'):
        """Store Dataset

        Args:
            dataset (Dict[str, np.ndarray]): dataset as generated by DataManager.build_dataset()
            settings (Dict[str, Any]): settings of the data generated
            directory (str): output folder
            filename (Optional[str], optional): Output filenam. Default is 'dataset.h5'.
        """
        data_fname = os.path.join(directory, filename)
        logger.info("Saving to {:s}".format(data_fname))
        compression = None
        dir_name = os.path.dirname(data_fname)
        settings_fname = os.path.join(directory, 'metadata.json')

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        try:
            with open(settings_fname, 'w') as fp:
                json.dump(obj=settings, fp=fp, cls=CustomEncoder, indent=2)

            with h5py.File(data_fname, 'w') as hf:
                for k, v in dataset.items():
                    if isinstance(k, str):
                        clean_k = k.replace('/', '_')
                        if clean_k is not k:
                            logger.warning(
                                'Warning: saving variable with name: {:s} as {:}'.format(k, clean_k))
                        else:
                            logger.info(
                                'Saving variable with name: {:s}'.format(clean_k))
                    else:
                        clean_k = k
                    hf.create_dataset(clean_k, data=v, compression=compression)
        except IOError as e:
            logger.error("Cannot open {:s} for writing.".format(data_fname))
            raise e

    @staticmethod
    def load_dataset(directory: str, filename: Optional[str] = 'dataset.h5', metadata_filename: Optional[str]
                     = 'metadata.json') -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Load Dataset

        Args:
            directory (str): output folder
            filename (Optional[str], optional): Output filenam. Default is 'dataset.h5'.

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, Any]]: Dataset as key-value pair and settings dict.
        """
        data_fname = os.path.join(directory, filename)
        dataset: Dict[str, Any] = {}
        if metadata_filename is not None:
            settings_fname = os.path.join(directory, metadata_filename)
        try:
            if metadata_filename is not None:
                with open(settings_fname, 'r') as fp:
                    settings = json.load(fp=fp)
            else:
                settings = {}
            with h5py.File(data_fname, 'r') as hf:
                for key in hf.keys():
                    dataset[key] = hf[key][()]
        except IOError as e:
            logger.error("Cannot open {:s} for reading.".format(data_fname))
            raise e
        return dataset, settings
