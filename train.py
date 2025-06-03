from config import Config
from dataset import get_dataloaders
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl
from torch import nn
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")
from pytorch_lightning.callbacks import ModelCheckpoint


class FeedForwardBlock(nn.Module):
    """Feed-forward block with dual linear transformations"""

    def __init__(self, feature_size):
        super(FeedForwardBlock, self).__init__()
        self.transform_1 = nn.Linear(feature_size, feature_size)
        self.transform_2 = nn.Linear(feature_size, feature_size)

    def forward(self, input_data):
        intermediate = F.relu(self.transform_1(input_data))
        final_output = self.transform_2(intermediate)
        return final_output


class ExerciseEmbeddingLayer(nn.Module):
    """Embedding layer for exercises with positional and categorical encodings"""

    def __init__(
        self, exercise_vocab_size, category_vocab_size, dimension_size, max_length
    ):
        super(ExerciseEmbeddingLayer, self).__init__()
        self.dimension_size = dimension_size
        self.max_length = max_length

        # Create embedding matrices
        self.exercise_embeddings = nn.Embedding(exercise_vocab_size, dimension_size)
        self.category_embeddings = nn.Embedding(category_vocab_size, dimension_size)
        self.position_embeddings = nn.Embedding(max_length, dimension_size)

    def forward(self, exercise_tokens, category_tokens):
        # Extract embeddings
        exercise_vectors = self.exercise_embeddings(exercise_tokens)
        category_vectors = self.category_embeddings(category_tokens)

        # Generate positional embeddings with proper batch size handling
        batch_size = exercise_tokens.size(0)
        position_indices = (
            torch.arange(self.max_length, device=exercise_tokens.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        position_vectors = self.position_embeddings(position_indices)

        # Aggregate embeddings
        aggregated_embeddings = position_vectors + category_vectors + exercise_vectors
        return aggregated_embeddings


class ResponseEmbeddingLayer(nn.Module):
    """Embedding layer for responses with positional information"""

    def __init__(self, response_vocab_size, dimension_size, max_length):
        super(ResponseEmbeddingLayer, self).__init__()
        self.dimension_size = dimension_size
        self.max_length = max_length

        # Initialize embedding components
        self.response_embeddings = nn.Embedding(response_vocab_size, dimension_size)
        self.temporal_projection = nn.Linear(1, dimension_size, bias=False)
        self.position_embeddings = nn.Embedding(max_length, dimension_size)

    def forward(self, response_tokens):
        # Get response embeddings
        response_vectors = self.response_embeddings(response_tokens)

        # Create positional embeddings with proper batch size handling
        batch_size = response_tokens.size(0)
        position_indices = (
            torch.arange(self.max_length, device=response_tokens.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        position_vectors = self.position_embeddings(position_indices)

        # Combine embeddings
        combined_vectors = position_vectors + response_vectors
        return combined_vectors


class StackedMultiHeadTransformer(nn.Module):
    """Transformer with multiple stacked attention layers"""

    def __init__(
        self,
        num_layers,
        dimension_size,
        attention_heads,
        sequence_length,
        heads_per_layer=1,
        dropout_prob=0.0,
    ):
        super(StackedMultiHeadTransformer, self).__init__()

        # Store configuration
        self.num_layers = num_layers
        self.heads_per_layer = heads_per_layer
        self.dimension_size = dimension_size

        # Initialize layer normalization
        self.normalization_layer = nn.LayerNorm(dimension_size)

        # Build attention architecture
        self.attention_modules = self._build_attention_architecture(
            num_layers, heads_per_layer, dimension_size, attention_heads, dropout_prob
        )

        # Create feed-forward blocks
        self.feedforward_modules = nn.ModuleList(
            [FeedForwardBlock(dimension_size) for _ in range(num_layers)]
        )

        # Setup attention masking
        self.causal_mask = self._setup_causal_masking(sequence_length)

    def _build_attention_architecture(
        self, num_layers, heads_per_layer, dimension_size, attention_heads, dropout_prob
    ):
        """Construct the multi-layer attention architecture"""
        architecture = nn.ModuleList()
        for _ in range(num_layers):
            layer_heads = nn.ModuleList()
            for _ in range(heads_per_layer):
                attention_head = nn.MultiheadAttention(
                    embed_dim=dimension_size,
                    num_heads=attention_heads,
                    dropout=dropout_prob,
                )
                layer_heads.append(attention_head)
            architecture.append(layer_heads)
        return architecture

    def _setup_causal_masking(self, sequence_length):
        """Create causal attention mask"""
        mask_matrix = torch.triu(
            torch.ones(sequence_length, sequence_length), diagonal=1
        )
        return mask_matrix.to(dtype=torch.bool)

    def _execute_attention_operation(
        self, query_tensor, key_tensor, value_tensor, layer_index, head_index
    ):
        """Execute single attention operation with normalization"""
        # Apply layer normalization
        normalized_query = self.normalization_layer(query_tensor)
        normalized_key = self.normalization_layer(key_tensor)
        normalized_value = self.normalization_layer(value_tensor)

        # Perform attention computation with proper device placement
        attention_result, _ = self.attention_modules[layer_index][head_index](
            query=normalized_query.permute(1, 0, 2),
            key=normalized_key.permute(1, 0, 2),
            value=normalized_value.permute(1, 0, 2),
            attn_mask=self.causal_mask.to(query_tensor.device),
        )

        return attention_result.permute(1, 0, 2)

    def forward(
        self,
        query_input,
        key_input,
        value_input,
        external_encoder_output=None,
        cross_attention_layer=None,
    ):
        # Initialize working tensors
        current_query, current_key, current_value = query_input, key_input, value_input

        # Process through layers
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.heads_per_layer):
                # Apply attention mechanism
                attention_output = self._execute_attention_operation(
                    current_query, current_key, current_value, layer_idx, head_idx
                )

                # Handle cross-attention if specified
                if (
                    external_encoder_output is not None
                    and head_idx == cross_attention_layer
                ):
                    assert (
                        cross_attention_layer <= head_idx
                    ), "cross attention layer should be less than heads per layer and positive integer"
                    current_key = current_value = external_encoder_output
                    current_query = current_query + attention_output
                else:
                    current_query = current_query + attention_output
                    current_key = current_key + attention_output
                    current_value = current_value + attention_output

            # Apply feed-forward transformation
            normalized_output = self.normalization_layer(attention_output)
            feedforward_result = self.feedforward_modules[layer_idx](normalized_output)
            attention_output = feedforward_result + attention_output

        return attention_output


class SAINTPlusLightningModule(pl.LightningModule):
    """Lightning module for SAINT+ knowledge tracing model"""

    def __init__(self):
        super(SAINTPlusLightningModule, self).__init__()

        # Initialize loss computation
        self.loss_function = nn.BCEWithLogitsLoss()

        # Setup encoder transformer
        self.encoder_transformer = StackedMultiHeadTransformer(
            num_layers=Config.NUM_DECODER,
            dimension_size=Config.EMBED_DIMS,
            attention_heads=Config.DEC_HEADS,
            sequence_length=Config.MAX_SEQ,
            heads_per_layer=1,
            dropout_prob=0.0,
        )

        # Setup decoder transformer
        self.decoder_transformer = StackedMultiHeadTransformer(
            num_layers=Config.NUM_ENCODER,
            dimension_size=Config.EMBED_DIMS,
            attention_heads=Config.ENC_HEADS,
            sequence_length=Config.MAX_SEQ,
            heads_per_layer=2,
            dropout_prob=0.0,
        )

        # Initialize embedding components
        self.exercise_embedding_layer = ExerciseEmbeddingLayer(
            exercise_vocab_size=Config.TOTAL_EXE,
            category_vocab_size=Config.TOTAL_CAT,
            dimension_size=Config.EMBED_DIMS,
            max_length=Config.MAX_SEQ,
        )

        self.response_embedding_layer = ResponseEmbeddingLayer(
            response_vocab_size=3,
            dimension_size=Config.EMBED_DIMS,
            max_length=Config.MAX_SEQ,
        )

        # Additional components
        self.temporal_embedding = nn.Linear(1, Config.EMBED_DIMS)
        self.classification_head = nn.Linear(Config.EMBED_DIMS, 1)

        # Training state tracking
        self.training_outputs = []
        self.validation_outputs = []

    def _prepare_encoder_input(self, input_batch):
        """Prepare encoder input embeddings"""
        return self.exercise_embedding_layer(
            exercise_tokens=input_batch["input_ids"],
            category_tokens=input_batch["input_cat"],
        )

    def _prepare_decoder_input(self, response_batch, input_batch):
        """Prepare decoder input embeddings with temporal information"""
        # Get basic response embeddings
        response_embeddings = self.response_embedding_layer(response_batch)

        # Process temporal features
        temporal_features = input_batch["input_rtime"].unsqueeze(-1).float()
        temporal_embeddings = self.temporal_embedding(temporal_features)

        # Combine embeddings
        enhanced_embeddings = response_embeddings + temporal_embeddings
        return enhanced_embeddings

    def forward(self, input_batch, response_batch):
        # Process encoder inputs
        encoder_embeddings = self._prepare_encoder_input(input_batch)

        # Process decoder inputs
        decoder_embeddings = self._prepare_decoder_input(response_batch, input_batch)

        # Apply encoder transformation
        encoder_output = self.encoder_transformer(
            query_input=encoder_embeddings,
            key_input=encoder_embeddings,
            value_input=encoder_embeddings,
        )

        # Apply decoder transformation with cross-attention
        decoder_output = self.decoder_transformer(
            query_input=decoder_embeddings,
            key_input=decoder_embeddings,
            value_input=decoder_embeddings,
            external_encoder_output=encoder_output,
            cross_attention_layer=1,
        )

        # Generate predictions
        logits = self.classification_head(decoder_output)
        return logits.squeeze()

    def configure_optimizers(self):
        """Configure optimizer for training"""
        return torch.optim.Adam(self.parameters())

    def _compute_metrics_and_log(
        self, predictions, labels, target_mask, loss_value, step_type, batch_idx
    ):
        """Compute metrics and handle logging"""
        # Apply masking and activation
        masked_predictions = torch.masked_select(predictions, target_mask)
        activated_predictions = torch.sigmoid(masked_predictions)
        masked_labels = torch.masked_select(labels, target_mask)

        # Log loss
        self.log(f"{step_type}_loss", loss_value, on_step=True, prog_bar=True)

        # Store outputs for epoch-end computation
        output_dict = {
            "outs": activated_predictions.detach().cpu(),
            "labels": masked_labels.detach().cpu(),
        }

        if step_type == "train":
            self.training_outputs.append(output_dict)
        else:
            self.validation_outputs.append(output_dict)

        return {
            "loss": loss_value,
            "outs": activated_predictions,
            "labels": masked_labels,
        }

    def training_step(self, batch, batch_idx):
        """Execute single training step"""
        input_data, label_data = batch
        mask = input_data["input_ids"] != 0

        # Forward pass
        model_output = self(input_data, label_data)

        # Periodic label analysis
        if batch_idx % 10 == 0:
            self._analyze_labels(label_data, mask)

        # Compute loss
        step_loss = self.loss_function(model_output.float(), label_data.float())

        # Process and log metrics
        return self._compute_metrics_and_log(
            model_output, label_data, mask, step_loss, "train", batch_idx
        )

    def _analyze_labels(self, labels, mask):
        """Analyze label distribution for debugging"""
        valid_labels = torch.masked_select(labels, mask)

        # Display sample labels
        sample_labels = valid_labels[:20].cpu().numpy()
        print(f"\nStep {self.global_step} Label samples: {sample_labels.tolist()}")

        # Log statistics
        statistics = {
            "total_samples": valid_labels.shape[0],
            "positive_ratio": torch.mean(valid_labels.float()).item(),
        }
        self.logger.experiment.add_text(
            "Label_Statistics", str(statistics), global_step=self.global_step
        )

    def on_train_epoch_end(self):
        """Process training epoch end"""
        self._process_epoch_end(self.training_outputs, "train")
        self.training_outputs.clear()

    def validation_step(self, batch, batch_idx):
        """Execute single validation step"""
        input_data, label_data = batch
        mask = input_data["input_ids"] != 0

        # Forward pass
        model_output = self(input_data, label_data)

        # Compute loss
        step_loss = self.loss_function(model_output.float(), label_data.float())

        # Process and log metrics
        return self._compute_metrics_and_log(
            model_output, label_data, mask, step_loss, "val", batch_idx
        )

    def on_validation_epoch_end(self):
        """Process validation epoch end"""
        self._process_epoch_end(self.validation_outputs, "val")
        self.validation_outputs.clear()

    def _process_epoch_end(self, epoch_outputs, phase):
        """Process epoch end for both training and validation"""
        if len(epoch_outputs) > 0:
            # Aggregate predictions and labels
            all_predictions = np.concatenate(
                [output["outs"].numpy() for output in epoch_outputs]
            ).reshape(-1)

            all_labels = np.concatenate(
                [output["labels"].numpy() for output in epoch_outputs]
            ).reshape(-1)

            # Compute AUC score
            auc_score = roc_auc_score(all_labels, all_predictions)

            # Log results
            self.print(f"{phase} auc", auc_score)
            self.log(f"{phase}_auc", auc_score)


def execute_training_pipeline():
    """Main training pipeline execution"""
    # Load data
    training_loader, validation_loader = get_dataloaders()

    # Initialize model
    saint_model = SAINTPlusLightningModule()

    # Setup logging
    tensorboard_logger = TensorBoardLogger("logs/", name="saint_plus_experiment")

    # Configure checkpointing
    checkpoint_handler = ModelCheckpoint(
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        dirpath="./saved_models",
        filename="best_model",
    )

    # Setup trainer
    model_trainer = pl.Trainer(
        logger=tensorboard_logger,
        accelerator="auto",
        devices="auto",
        max_epochs=2,
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=[checkpoint_handler],
    )

    # Execute training
    model_trainer.fit(
        model=saint_model,
        train_dataloaders=training_loader,
        val_dataloaders=validation_loader,
    )


if __name__ == "__main__":
    execute_training_pipeline()
