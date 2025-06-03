from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from config import Config
from sklearn.model_selection import train_test_split
import gc


class KnowledgeTracingDataset(Dataset):
    """Dataset class for knowledge tracing with sequence processing"""

    def __init__(self, user_sequences, maximum_sequence_length):
        super(KnowledgeTracingDataset, self).__init__()
        self.user_sequences = user_sequences
        self.maximum_sequence_length = maximum_sequence_length
        self.processed_sequences = []
        self._process_all_sequences()

    def _process_all_sequences(self):
        """Process all user sequences into training samples"""
        for user_index in self.user_sequences.index:
            sequence_data = self.user_sequences[user_index]
            exercise_sequence, response_sequence, time_sequence, category_sequence = (
                sequence_data
            )

            if self._should_split_sequence(exercise_sequence):
                self._split_long_sequence(
                    exercise_sequence,
                    response_sequence,
                    time_sequence,
                    category_sequence,
                )
            elif self._is_valid_sequence_length(exercise_sequence):
                self._add_sequence(
                    exercise_sequence,
                    response_sequence,
                    time_sequence,
                    category_sequence,
                )

    def _should_split_sequence(self, exercise_sequence):
        """Check if sequence needs to be split due to length"""
        return len(exercise_sequence) > self.maximum_sequence_length

    def _is_valid_sequence_length(self, exercise_sequence):
        """Check if sequence has valid length for training"""
        sequence_length = len(exercise_sequence)
        return self.maximum_sequence_length > sequence_length > 50

    def _split_long_sequence(self, exercises, responses, times, categories):
        """Split long sequence into multiple training samples"""
        total_length = len(exercises)
        num_splits = (
            total_length + self.maximum_sequence_length - 1
        ) // self.maximum_sequence_length

        for split_index in range(num_splits):
            start_idx = split_index * self.maximum_sequence_length
            end_idx = start_idx + self.maximum_sequence_length

            self._add_sequence(
                exercises[start_idx:end_idx],
                responses[start_idx:end_idx],
                times[start_idx:end_idx],
                categories[start_idx:end_idx],
            )

    def _add_sequence(self, exercise_ids, answer_ids, elapsed_times, category_ids):
        """Add processed sequence to dataset"""
        self.processed_sequences.append(
            (exercise_ids, answer_ids, elapsed_times, category_ids)
        )

    def __len__(self):
        """Return total number of processed sequences"""
        return len(self.processed_sequences)

    def __getitem__(self, sequence_index):
        """Get a single training sample with proper padding"""
        exercise_ids, response_ids, elapsed_times, category_ids = (
            self.processed_sequences[sequence_index]
        )
        actual_length = len(exercise_ids)

        # Initialize padded arrays
        padded_exercises = self._create_padded_array(
            self.maximum_sequence_length, dtype=int
        )
        padded_responses = self._create_padded_array(
            self.maximum_sequence_length, dtype=int
        )
        padded_times = self._create_padded_array(
            self.maximum_sequence_length, dtype=int
        )
        padded_categories = self._create_padded_array(
            self.maximum_sequence_length, dtype=int
        )

        # Fill arrays with actual data
        if actual_length < self.maximum_sequence_length:
            self._fill_arrays_right_aligned(
                padded_exercises,
                padded_responses,
                padded_times,
                padded_categories,
                exercise_ids,
                response_ids,
                elapsed_times,
                category_ids,
                actual_length,
            )
        else:
            self._fill_arrays_truncated(
                padded_exercises,
                padded_responses,
                padded_times,
                padded_categories,
                exercise_ids,
                response_ids,
                elapsed_times,
                category_ids,
            )

        # Create shifted time array for input
        shifted_times = self._create_shifted_time_array(padded_times)

        # Prepare input dictionary
        input_features = self._create_input_dictionary(
            padded_exercises, shifted_times, padded_categories
        )

        return input_features, padded_responses

    def _create_padded_array(self, length, dtype=int):
        """Create zero-padded array of specified length and type"""
        return np.zeros(length, dtype=dtype)

    def _fill_arrays_right_aligned(
        self,
        padded_ex,
        padded_resp,
        padded_time,
        padded_cat,
        exercise_ids,
        response_ids,
        elapsed_times,
        category_ids,
        length,
    ):
        """Fill arrays with right alignment (padding on left)"""
        padded_ex[-length:] = exercise_ids
        padded_resp[-length:] = response_ids
        padded_time[-length:] = elapsed_times
        padded_cat[-length:] = category_ids

    def _fill_arrays_truncated(
        self,
        padded_ex,
        padded_resp,
        padded_time,
        padded_cat,
        exercise_ids,
        response_ids,
        elapsed_times,
        category_ids,
    ):
        """Fill arrays with truncation from the end"""
        max_len = self.maximum_sequence_length
        padded_ex[:] = exercise_ids[-max_len:]
        padded_resp[:] = response_ids[-max_len:]
        padded_time[:] = elapsed_times[-max_len:]
        padded_cat[:] = category_ids[-max_len:]

    def _create_shifted_time_array(self, time_array):
        """Create time array shifted by one position for input"""
        shifted_array = np.zeros(self.maximum_sequence_length, dtype=int)
        shifted_array = np.insert(time_array, 0, 0)
        shifted_array = np.delete(shifted_array, -1)
        return shifted_array.astype(int)

    def _create_input_dictionary(self, exercise_array, time_array, category_array):
        """Create input dictionary with proper keys"""
        return {
            "input_ids": exercise_array,
            "input_rtime": time_array,
            "input_cat": category_array,
        }


class DataLoaderFactory:
    """Factory class for creating data loaders"""

    def __init__(self):
        self.data_type_specifications = self._define_data_types()

    def _define_data_types(self):
        """Define data types for efficient memory usage"""
        return {
            "timestamp": "int64",
            "user_id": "int32",
            "content_id": "int16",
            "answered_correctly": "int8",
            "content_type_id": "int8",
            "prior_question_elapsed_time": "float32",
            "task_container_id": "int16",
        }

    def create_data_loaders(self):
        """Create training and validation data loaders"""
        print("Starting data loading process...")

        # Load and preprocess data
        processed_data = self._load_and_preprocess_data()

        # Split into train and validation
        training_data, validation_data = self._split_data(processed_data)

        # Create datasets
        train_dataset = self._create_dataset(training_data, "training")
        val_dataset = self._create_dataset(validation_data, "validation")

        # Create data loaders
        training_loader = self._create_loader(train_dataset, shuffle=True)
        validation_loader = self._create_loader(val_dataset, shuffle=False)

        # Cleanup memory
        self._cleanup_datasets(train_dataset, val_dataset)

        return training_loader, validation_loader

    def _load_and_preprocess_data(self):
        """Load CSV data and apply preprocessing"""
        print("Loading CSV data...")

        # Load data with specified columns and types
        raw_dataframe = pd.read_csv(
            Config.TRAIN_FILE,
            usecols=[1, 2, 3, 4, 5, 7, 8],
            dtype=self.data_type_specifications,
            nrows=90e6,
        )

        print(f"Initial dataframe shape: {raw_dataframe.shape}")

        # Apply data filtering and preprocessing
        processed_dataframe = self._apply_data_preprocessing(raw_dataframe)

        # Group data by user
        grouped_data = self._group_data_by_user(processed_dataframe)

        # Clean up memory
        del raw_dataframe, processed_dataframe
        gc.collect()

        return grouped_data

    def _apply_data_preprocessing(self, dataframe):
        """Apply preprocessing steps to the dataframe"""
        # Filter by content type and create a copy to avoid warnings
        filtered_df = dataframe[dataframe.content_type_id == 0].copy()

        # Handle missing elapsed time values
        filtered_df = self._process_elapsed_time(filtered_df)

        # Sort by timestamp
        sorted_df = filtered_df.sort_values(["timestamp"], ascending=True).reset_index(
            drop=True
        )

        # Log statistics
        num_unique_skills = sorted_df.content_id.nunique()
        print(f"Number of unique skills: {num_unique_skills}")
        print(f"Shape after preprocessing: {sorted_df.shape}")

        return sorted_df

    def _process_elapsed_time(self, dataframe):
        """Process elapsed time column"""
        # Fill missing values
        dataframe.loc[:, "prior_question_elapsed_time"] = dataframe[
            "prior_question_elapsed_time"
        ].fillna(0)

        # Convert to seconds
        dataframe.loc[:, "prior_question_elapsed_time"] = (
            dataframe["prior_question_elapsed_time"] / 1000
        )

        # Convert to integer
        dataframe.loc[:, "prior_question_elapsed_time"] = dataframe[
            "prior_question_elapsed_time"
        ].astype(int)

        return dataframe

    def _group_data_by_user(self, dataframe):
        """Group data by user and create sequences"""
        print("Grouping data by users...")

        grouped_sequences = (
            dataframe[
                [
                    "user_id",
                    "content_id",
                    "answered_correctly",
                    "prior_question_elapsed_time",
                    "task_container_id",
                ]
            ]
            .groupby("user_id")
            .apply(self._extract_user_sequence)
        )

        return grouped_sequences

    def _extract_user_sequence(self, user_data):
        """Extract sequence data for a single user"""
        return (
            user_data.content_id.values,
            user_data.answered_correctly.values,
            user_data.prior_question_elapsed_time.values,
            user_data.task_container_id.values,
        )

    def _split_data(self, grouped_data):
        """Split data into training and validation sets"""
        print("Splitting data into train and validation sets...")

        training_sequences, validation_sequences = train_test_split(
            grouped_data, test_size=0.2
        )

        print(f"Training set size: {training_sequences.shape}")
        print(f"Validation set size: {validation_sequences.shape}")

        return training_sequences, validation_sequences

    def _create_dataset(self, sequence_data, dataset_type):
        """Create dataset instance from sequence data"""
        return KnowledgeTracingDataset(
            user_sequences=sequence_data, maximum_sequence_length=Config.MAX_SEQ
        )

    def _create_loader(self, dataset, shuffle=True):
        """Create data loader with specified configuration"""
        return DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            persistent_workers=True,
            num_workers=8,
            shuffle=shuffle,
        )

    def _cleanup_datasets(self, train_dataset, val_dataset):
        """Clean up dataset objects to free memory"""
        del train_dataset, val_dataset
        gc.collect()

    def _save_dataset_samples(self, dataset, output_filename, sample_count=500):
        """Save sample data for inspection (optional utility method)"""
        with open(output_filename, "w", encoding="utf-8") as output_file:
            for sample_idx in range(min(sample_count, len(dataset))):
                input_data, label_data = dataset[sample_idx]

                output_file.write(f"Sample {sample_idx}:\n")
                output_file.write(f"Exercise IDs: {input_data['input_ids']}\n")
                output_file.write(f"Time sequence: {input_data['input_rtime']}\n")
                output_file.write(f"Categories: {input_data['input_cat']}\n")
                output_file.write(f"Labels: {label_data}\n")
                output_file.write("-" * 50 + "\n")


def get_dataloaders():
    """Main function to get training and validation data loaders"""
    factory = DataLoaderFactory()
    return factory.create_data_loaders()
