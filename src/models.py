import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pooling import TemporalAttentionPooling, TemporalAvgPooling, TemporalLastPooling, TemporalMaxPooling


# =============================================
#               Transformer Model
# =============================================

class CreditsEmbedder(nn.Module):
    def __init__(self, features, embedding_projections):
        super(CreditsEmbedder, self).__init__()

        self._credits_cat_embeddings = nn.ModuleList(
            [self._create_embedding_projection(*embedding_projections[feature])
             for feature in features]
        )


    def forward(self, features):
        batch_size = features[0].shape[0]
        seq_len = features[0].shape[1]
        embeddings = [embedding(features[i]) for i, embedding in enumerate(
            self._credits_cat_embeddings)]

        concated_embeddings = torch.cat(embeddings, dim=-1)
        return concated_embeddings

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)


class TransformerCreditsModel(nn.Module):
    def __init__(
        self,
        features,
        embedding_projections,
        dim_feedforward=128,
        nhead=3,
        nlayers=4,
        p_dropout=0.1,
        top_classifier_units=32
    ):
        super(TransformerCreditsModel, self).__init__()
        self._credits_embedder = CreditsEmbedder(
            features, embedding_projections)

        self._d_model = sum([embedding_projections[x][1] for x in features])

        self._encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=p_dropout
        )

        self._transformer_enc = nn.TransformerEncoder(
            self._encoder_layer,
            num_layers=nlayers
        )

        self._top_classifier = nn.Linear(in_features=2 * self._d_model,
                                         out_features=top_classifier_units)
        self._intermediate_activation = nn.ReLU()
        self._head = nn.Linear(
            in_features=top_classifier_units, out_features=1)

    def forward(self, features):
        emb = self._credits_embedder(features)
        out = self._transformer_enc(emb.permute(1, 0, 2))
        # poolings
        out_max_pool = out.permute(1, 0, 2).max(dim=1)[0]
        out_avg_pool = out.permute(1, 0, 2).sum(
            dim=1) / out.permute(1, 0, 2).shape[1]

        combined_input = torch.cat([out_max_pool, out_avg_pool], dim=-1)
        classification_hidden = self._top_classifier(combined_input)

        activation = self._intermediate_activation(classification_hidden)
        logits = self._head(activation)

        return logits.squeeze(1)


# =================================================
#               RNN-based Models
# =================================================

def get_rnn_model(rnn_type, **kwargs):
    model_dict = {
        "LSTM": nn.LSTM,
        "GRU": nn.GRU
    }

    return model_dict[rnn_type](**kwargs)


class CreditsRNN(nn.Module):
    def __init__(self, features, embedding_projections,
                 rnn_type="GRU", rnn_units=128, rnn_num_layers=1, top_classifier_units=32):
        super(CreditsRNN, self).__init__()
        self._credits_cat_embeddings = nn.ModuleList(
            [self._create_embedding_projection(*embedding_projections[feature])
             for feature in features]
        )

        assert rnn_type in ["GRU", "LSTM"]
        self._rnn = get_rnn_model(
            rnn_type,
            input_size=sum([embedding_projections[x][1] for x in features]),
            hidden_size=rnn_units,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=True
        )

        self._hidden_size = rnn_units
        self._top_classifier = nn.Linear(in_features=4*rnn_units,
                                         out_features=top_classifier_units)
        self._intermediate_activation = nn.ReLU()
        self._head = nn.Linear(in_features=top_classifier_units,
                               out_features=1)

    def forward(self, features):
        batch_size = features[0].shape[0]

        embeddings = [embedding(features[i]) for i, embedding in enumerate(
            self._credits_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)

        hidden_states_seq, _ = self._rnn(concated_embeddings)

        # batch_size, seq_leq, 2 * hidden_state
        out_max_pool = hidden_states_seq.max(dim=1)[0]
        out_avg_pool = hidden_states_seq.sum(
            dim=1) / hidden_states_seq.shape[1]

        combined_input = torch.cat([out_max_pool, out_avg_pool], dim=-1)

        classification_hidden = self._top_classifier(combined_input)
        activation = self._intermediate_activation(classification_hidden)
        logits = self._head(activation)

        return logits.squeeze(1)

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(
            num_embeddings=cardinality+add_missing,
            embedding_dim=embed_size,
            padding_idx=padding_idx
        )

# =============================================================
#               Advanced RNN models
# =============================================================


class CreditsAdvancedRNN(nn.Module):
    """ @TODO Docs."""
    def __init__(
        self, 
        features, 
        embedding_projections,
        rnn_type="GRU", 
        rnn_units=128, 
        rnn_num_layers=1, 
        top_classifier_units=64,
    ):
        super(CreditsAdvancedRNN, self).__init__()
        self._credits_cat_embeddings = nn.ModuleList(
            [self._create_embedding_projection(*embedding_projections[feature])
             for feature in features]
        )

        assert rnn_type in ["GRU", "LSTM"]
        self._rnn = get_rnn_model(
            rnn_type,
            input_size=sum([embedding_projections[x][1] for x in features]),
            hidden_size=rnn_units,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=True
        )
        self._spatial_dropout = nn.Dropout2d(0.1)
        
        self._poolings = nn.ModuleDict({
            "max" : TemporalMaxPooling(), "avg": TemporalAvgPooling(),
            "last": TemporalLastPooling(), "attn": TemporalAttentionPooling(2 * rnn_units)
        })

        self._hidden_size = rnn_units
        self._top_classifier = nn.Linear(in_features=8*rnn_units,
                                         out_features=top_classifier_units)
        self._intermediate_activation = nn.ReLU()
        self._head = nn.Linear(in_features=top_classifier_units,
                               out_features=1)

    def forward(self, features):
        batch_size = features[0].shape[0]

        embeddings = [embedding(features[i]) for i, embedding in enumerate(
            self._credits_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)
        # spatial dropout
        concated_embeddings = concated_embeddings.permute(0, 2, 1).unsqueeze(3)
        dropout_embeddings = self._spatial_dropout(concated_embeddings)
        dropout_embeddings = dropout_embeddings.squeeze(3).permute(0, 2, 1)

        hidden_states_seq, _ = self._rnn(dropout_embeddings)
        # spatial dropout
        hidden_states_seq = self._spatial_dropout(hidden_states_seq.permute(0, 2, 1).unsqueeze(3))
        hidden_states_seq = hidden_states_seq.squeeze(3).permute(0, 2, 1)
        
        # [batch_size, seq_leq, 2 * hidden_state]
        poolings_output = [pooling(hidden_states_seq) for _, pooling in self._poolings.items()]  
        combined_input = torch.cat(poolings_output, dim=-1)

        classification_hidden = self._top_classifier(combined_input)
        activation = self._intermediate_activation(classification_hidden)
        logits = self._head(activation)

        return logits.squeeze(1)

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(
            num_embeddings=cardinality+add_missing,
            embedding_dim=embed_size,
            padding_idx=padding_idx
        )
