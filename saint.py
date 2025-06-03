import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class FeedForwardNetwork(nn.Module):
    """Feed-forward network with ReLU activation"""

    def __init__(self, feature_dim):
        super(FeedForwardNetwork, self).__init__()
        self.first_linear = nn.Linear(feature_dim, feature_dim)
        self.second_linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, input_tensor):
        activated = F.relu(self.first_linear(input_tensor))
        output = self.second_linear(activated)
        return output


class ExerciseEncoder(nn.Module):
    """Encoder for exercise embeddings with position and category information"""

    def __init__(self, exercise_count, category_count, embedding_dim, sequence_length):
        super(ExerciseEncoder, self).__init__()
        self.embedding_dimension = embedding_dim
        self.max_sequence_length = sequence_length

        # Initialize embedding layers
        self.exercise_embedding = nn.Embedding(exercise_count, embedding_dim)
        self.category_embedding = nn.Embedding(category_count, embedding_dim)
        self.positional_embedding = nn.Embedding(sequence_length, embedding_dim)

    def forward(self, exercise_ids, category_ids):
        # Get exercise embeddings
        exercise_emb = self.exercise_embedding(exercise_ids)
        # Get category embeddings
        category_emb = self.category_embedding(category_ids)

        # Create position sequence with proper batch size handling
        batch_size = exercise_ids.size(0)
        position_sequence = (
            torch.arange(self.max_sequence_length, device=exercise_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        positional_emb = self.positional_embedding(position_sequence)

        # Combine all embeddings
        combined_embedding = positional_emb + category_emb + exercise_emb
        return combined_embedding


class ResponseDecoder(nn.Module):
    """Decoder for response embeddings with positional encoding"""

    def __init__(self, response_count, embedding_dim, sequence_length):
        super(ResponseDecoder, self).__init__()
        self.embedding_dimension = embedding_dim
        self.max_sequence_length = sequence_length

        # Initialize embedding layers
        self.response_embedding = nn.Embedding(response_count, embedding_dim)
        self.temporal_embedding = nn.Linear(1, embedding_dim, bias=False)
        self.positional_embedding = nn.Embedding(sequence_length, embedding_dim)

    def forward(self, response_sequence):
        # Get response embeddings
        response_emb = self.response_embedding(response_sequence)

        # Create position sequence with proper batch size handling
        batch_size = response_sequence.size(0)
        position_sequence = (
            torch.arange(self.max_sequence_length, device=response_sequence.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        positional_emb = self.positional_embedding(position_sequence)

        # Combine embeddings
        combined_embedding = positional_emb + response_emb
        return combined_embedding


class MultiLayerAttentionStack(nn.Module):
    """Multi-layer multi-head attention with stacking capability"""

    def __init__(
        self,
        layer_count,
        embedding_dim,
        head_count,
        seq_length,
        multihead_count=1,
        dropout_rate=0.0,
    ):
        super(MultiLayerAttentionStack, self).__init__()

        self.layer_count = layer_count
        self.multihead_count = multihead_count
        self.embedding_dim = embedding_dim

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Create multi-head attention layers
        self.attention_stack = self._create_attention_layers(
            layer_count, multihead_count, embedding_dim, head_count, dropout_rate
        )

        # Feed-forward networks for each layer
        self.ffn_stack = nn.ModuleList(
            [FeedForwardNetwork(embedding_dim) for _ in range(layer_count)]
        )

        # Create attention mask
        self.attention_mask = self._create_causal_mask(seq_length)

    def _create_attention_layers(
        self, layer_count, multihead_count, embedding_dim, head_count, dropout_rate
    ):
        """Helper method to create attention layer structure"""
        return nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.MultiheadAttention(
                            embed_dim=embedding_dim,
                            num_heads=head_count,
                            dropout=dropout_rate,
                        )
                        for _ in range(multihead_count)
                    ]
                )
                for _ in range(layer_count)
            ]
        )

    def _create_causal_mask(self, seq_length):
        """Create upper triangular mask for causal attention"""
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
        return mask.to(dtype=torch.bool)

    def _apply_attention_layer(self, query, key, value, layer_idx, head_idx):
        """Apply single attention layer with normalization"""
        normalized_q = self.layer_norm(query)
        normalized_k = self.layer_norm(key)
        normalized_v = self.layer_norm(value)

        # Apply attention with proper device placement
        attention_output, _ = self.attention_stack[layer_idx][head_idx](
            query=normalized_q.permute(1, 0, 2),
            key=normalized_k.permute(1, 0, 2),
            value=normalized_v.permute(1, 0, 2),
            attn_mask=self.attention_mask.to(query.device),
        )

        return attention_output.permute(1, 0, 2)

    def forward(self, input_q, input_k, input_v, encoder_output=None, break_layer=None):
        query, key, value = input_q, input_k, input_v

        for layer_idx in range(self.layer_count):
            for head_idx in range(self.multihead_count):
                # Apply attention
                attention_out = self._apply_attention_layer(
                    query, key, value, layer_idx, head_idx
                )

                # Handle encoder-decoder attention
                if encoder_output is not None and head_idx == break_layer:
                    assert (
                        break_layer <= head_idx
                    ), "break layer should be less than multihead layers and positive integer"
                    key = value = encoder_output
                    query = query + attention_out
                else:
                    query = query + attention_out
                    key = key + attention_out
                    value = value + attention_out

            # Apply feed-forward network
            normalized_attention = self.layer_norm(attention_out)
            ffn_result = self.ffn_stack[layer_idx](normalized_attention)
            attention_out = ffn_result + attention_out

        return attention_out


class PlusSAINTModule(nn.Module):
    """Enhanced SAINT model for knowledge tracing"""

    def __init__(self):
        super(PlusSAINTModule, self).__init__()

        # Initialize loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Create encoder stack
        self.encoder_attention = MultiLayerAttentionStack(
            layer_count=Config.NUM_DECODER,
            embedding_dim=Config.EMBED_DIMS,
            head_count=Config.DEC_HEADS,
            seq_length=Config.MAX_SEQ,
            multihead_count=1,
            dropout_rate=0.0,
        )

        # Create decoder stack
        self.decoder_attention = MultiLayerAttentionStack(
            layer_count=Config.NUM_ENCODER,
            embedding_dim=Config.EMBED_DIMS,
            head_count=Config.ENC_HEADS,
            seq_length=Config.MAX_SEQ,
            multihead_count=2,
            dropout_rate=0.0,
        )

        # Initialize embedding layers
        self.exercise_encoder = ExerciseEncoder(
            exercise_count=Config.TOTAL_EXE,
            category_count=Config.TOTAL_CAT,
            embedding_dim=Config.EMBED_DIMS,
            sequence_length=Config.MAX_SEQ,
        )

        self.response_decoder = ResponseDecoder(
            response_count=3,
            embedding_dim=Config.EMBED_DIMS,
            sequence_length=Config.MAX_SEQ,
        )

        # Time embedding and output layers
        self.time_embedding_layer = nn.Linear(1, Config.EMBED_DIMS)
        self.output_projection = nn.Linear(Config.EMBED_DIMS, 1)

    def _process_encoder_features(self, input_data):
        """Process encoder input features"""
        return self.exercise_encoder(
            exercise_ids=input_data["input_ids"], category_ids=input_data["input_cat"]
        )

    def _process_decoder_features(self, response_data, input_data):
        """Process decoder input features with time embedding"""
        decoder_emb = self.response_decoder(response_data)

        # Process elapsed time
        time_features = input_data["input_rtime"].unsqueeze(-1).float()
        time_emb = self.time_embedding_layer(time_features)

        # Combine decoder and time embeddings
        return decoder_emb + time_emb

    def forward(self, input_features, response_labels):
        # Process encoder features
        encoder_embeddings = self._process_encoder_features(input_features)

        # Process decoder features
        decoder_embeddings = self._process_decoder_features(
            response_labels, input_features
        )

        # Apply encoder attention
        encoder_representations = self.encoder_attention(
            input_k=encoder_embeddings,
            input_q=encoder_embeddings,
            input_v=encoder_embeddings,
        )

        # Apply decoder attention with encoder output
        decoder_representations = self.decoder_attention(
            input_k=decoder_embeddings,
            input_q=decoder_embeddings,
            input_v=decoder_embeddings,
            encoder_output=encoder_representations,
            break_layer=1,
        )

        # Generate final predictions
        predictions = self.output_projection(decoder_representations)
        return predictions.squeeze()
