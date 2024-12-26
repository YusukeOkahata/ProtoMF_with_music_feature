from abc import abstractmethod, ABC

import torch
import torch.nn as nn

from utilities.utils import general_weight_init

# 追加
import pandas as pd

class FeatureExtractor(nn.Module, ABC):
    """
    Abstract class representing one of the possible FeatureExtractor models. See also FeatureExtractorFactory.
    """

    def __init__(self):
        super().__init__()
        self.cumulative_loss = 0.
        self.name = "FeatureExtractor"

    def init_parameters(self):
        """
        Initial the Feature Extractor parameters
        """
        pass

    def get_and_reset_loss(self) -> float:
        """
        Reset the loss of the feature extractor and returns the computed value
        :return: loss of the feature extractor
        """
        loss = self.cumulative_loss
        self.cumulative_loss = 0.
        return loss

    @abstractmethod
    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        """
        Performs the feature extraction process of the object.
        """
        pass


class Embedding(FeatureExtractor):
    """
    FeatureExtractor that represents an object (item/user) only given by its embedding.
    """

    def __init__(self, n_objects: int, embedding_dim: int, max_norm: float = None, only_positive: bool = False):
        """
        Standard Embedding Layer
        :param n_objects: number of objects in the system (users or items)
        :param embedding_dim: embedding dimension
        :param max_norm: max norm of the l2 norm of the embeddings.
        :param only_positive: whether the embeddings can be only positive
        """
        super().__init__()
        self.n_objects = n_objects
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.only_positive = only_positive
        self.name = "Embedding"

        self.embedding_layer = nn.Embedding(self.n_objects, self.embedding_dim, max_norm=self.max_norm)
        print(f'Built Embedding model \n'
              f'- n_objects: {self.n_objects} \n'
              f'- embedding_dim: {self.embedding_dim} \n'
              f'- max_norm: {self.max_norm}\n'
              f'- only_positive: {self.only_positive}')

    def init_parameters(self):
        self.embedding_layer.apply(general_weight_init)

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        assert o_idxs is not None, f"Object Indexes not provided! ({self.name})"
        embeddings = self.embedding_layer(o_idxs)
        if self.only_positive:
            embeddings = torch.absolute(embeddings)
        return embeddings

# class EmbeddingWithAudioFeatures(FeatureExtractor):
#     def __init__(self, n_objects: int, embedding_dim: int, max_norm: float = None, audio_feature_dim: int = 13,
#                  out_dimension: int = None, use_bias: bool = False, audio_features_path: str = None):
#         super().__init__()

#         self.n_objects = n_objects
#         self.embedding_dim = embedding_dim
#         self.max_norm = max_norm
#         self.audio_feature_dim = audio_feature_dim
#         self.out_dimension = out_dimension
#         self.use_bias = use_bias
#         self.audio_features_path = audio_features_path
#         self.name = "EmbeddingWithAudioFeatures"

#         # Embedding layer from embedding_ext (Embedding class)
#         self.embedding_ext = Embedding(n_objects, embedding_dim, max_norm)  # Embeddingを使って初期化

#         # Additional layers for audio features
#         self.audio_feature_layer = nn.Linear(self.audio_feature_dim, self.embedding_dim)

#         # Initialize audio features
#         self.audio_features = None
#         if self.audio_features_path is not None:
#             try:
#                 import pandas as pd
#                 self.audio_features = pd.read_csv(self.audio_features_path)
#                 print(f"Audio features loaded successfully from: {self.audio_features_path}, shape: {self.audio_features.shape}")
#             except Exception as e:
#                 print(f"Error loading audio features: {e}")
#                 raise

#         print(f'Built EmbeddingWithAudioFeatures model \n'
#               f'- n_objects: {self.n_objects} \n'
#               f'- embedding_dim: {self.embedding_dim} \n'
#               f'- audio_feature_dim: {self.audio_feature_dim} \n'
#               f'- out_dimension: {self.out_dimension} \n'
#               f'- use_bias: {self.use_bias}')

#     def forward(self, o_idxs: torch.Tensor, audio_features: torch.Tensor = None) -> torch.Tensor:
#         """
#         :param o_idxs: Object indexes
#         :param audio_features: Audio features tensor
#         :return: Combined embedding
#         """
#         # Embedding for the objects from embedding_ext
#         embeddings = self.embedding_ext(o_idxs)  # embedding_ext (Embedding) のforwardを呼び出す

#         # Process audio features if provided
#         if audio_features is not None:
#             audio_embeddings = self.audio_feature_layer(audio_features)
#             embeddings += audio_embeddings  # Combine the embeddings

#         return embeddings

#     def _get_audio_features(self, o_idxs: torch.Tensor) -> torch.Tensor:
#         """
#         Get the audio features corresponding to the item indices.
#         """
#         if self.audio_features is None:
#             raise ValueError("Audio features are not initialized. Check the audio_features_path.")

#         try:
#             item_ids = o_idxs.cpu().numpy().flatten()  # インデックスをフラット化
#             audio_features = self.audio_features.iloc[item_ids].loc[:, 'feature_1':'feature_13'].values
#             audio_features_tensor = torch.tensor(audio_features, dtype=torch.float32).to(o_idxs.device)
#             return audio_features_tensor
#         except Exception as e:
#             print(f"Error fetching audio features for indices {o_idxs}: {e}")
#             raise

class EmbeddingWithAudioFeatures(FeatureExtractor):
    def __init__(self, n_objects: int, embedding_dim: int, max_norm: float = None, audio_feature_dim: int = 13,
                 out_dimension: int = None, use_bias: bool = False, audio_features_path: str = None):
        super().__init__()

        self.n_objects = n_objects
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.audio_feature_dim = audio_feature_dim
        self.out_dimension = out_dimension
        self.use_bias = use_bias
        self.audio_features_path = audio_features_path
        self.name = "EmbeddingWithAudioFeatures"

        # Embedding layer from embedding_ext (Embedding class)
        self.embedding_ext = Embedding(n_objects, embedding_dim, max_norm)  # Embeddingを使って初期化

        # Additional layers for audio features
        self.audio_feature_layer = nn.Linear(self.audio_feature_dim, self.embedding_dim)

        print(f'Built EmbeddingWithAudioFeatures model \n'
              f'- n_objects: {self.n_objects} \n'
              f'- embedding_dim: {self.embedding_dim} \n'
              f'- audio_feature_dim: {self.audio_feature_dim} \n'
              f'- out_dimension: {self.out_dimension} \n'
              f'- use_bias: {self.use_bias}')

    def forward(self, o_idxs: torch.Tensor, audio_features: torch.Tensor = None) -> torch.Tensor:
        """
        :param o_idxs: Object indexes
        :param audio_features: Audio features tensor
        :return: Combined embedding
        """
        # Embedding for the objects from embedding_ext
        embeddings = self.embedding_ext(o_idxs)  # embedding_ext (Embedding) のforwardを呼び出す

        # Process audio features if provided
        if audio_features is not None:
            audio_embeddings = self.audio_feature_layer(audio_features)
            embeddings += audio_embeddings  # Combine the embeddings

        return embeddings


    def _get_audio_features(self, o_idxs: torch.Tensor) -> torch.Tensor:
        """
        Get the audio features corresponding to the item indices.
        """
        try:
            # print(f"Object indices (o_idxs): {o_idxs}")
            
            item_ids = o_idxs.cpu().numpy().flatten()  # インデックスをフラット化
            audio_features = self.audio_features.iloc[item_ids].loc[:, 'feature_1':'feature_13'].values  # 'feature_1'から'feature_13'を抽出
            
            # # デバッグ用：取得したオーディオ特徴量の型とサンプルを表示
            # print(f"Fetched audio features (type): {type(audio_features)}")
            # print(f"Fetched audio features (dtype): {audio_features.dtype}")
            # print(f"Fetched audio features (sample): {audio_features[:5]}")  # 最初の5つを表示
            
            # 数値データのみを抽出して変換
            audio_features_tensor = torch.tensor(audio_features, dtype=torch.float32).to(o_idxs.device)
            
            return audio_features_tensor
        
        except Exception as e:
            print(f"Error fetching audio features for indices {o_idxs}: {e}")
            raise




class EmbeddingW(Embedding):
    """
    FeatureExtractor that places a linear projection after an embedding layer. Used for sharing weights.
    """

    def __init__(self, n_objects: int, embedding_dim: int, max_norm: float = None, out_dimension: int = None,
                 use_bias: bool = False):
        """
        :param n_objects: see Embedding
        :param embedding_dim: see Embedding
        :param max_norm: see Embedding
        :param out_dimension: Out dimension of the linear layer. If none, set to embedding_dim.
        :param use_bias: whether to use the bias in the linear layer.
        """
        super().__init__(n_objects, embedding_dim, max_norm)
        self.out_dimension = out_dimension
        self.use_bias = use_bias

        if self.out_dimension is None:
            self.out_dimension = embedding_dim

        self.name = 'EmbeddingW'
        self.linear_layer = nn.Linear(self.embedding_dim, self.out_dimension, bias=self.use_bias)

        print(f'Built Embeddingw model \n'
              f'- out_dimension: {self.out_dimension} \n'
              f'- use_bias: {self.use_bias} \n')

    def init_parameters(self):
        super().init_parameters()
        self.linear_layer.apply(general_weight_init)

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        o_embed = super().forward(o_idxs)
        return self.linear_layer(o_embed)





class EmbeddingWWithAudioFeatures(EmbeddingWithAudioFeatures):
    def __init__(self, n_objects: int, embedding_dim: int, audio_feature_dim: int = 13,
                 max_norm: float = None, out_dimension: int = None, use_bias: bool = False,
                 audio_features_path: str = None):
        super().__init__(n_objects, embedding_dim, max_norm, audio_feature_dim, out_dimension, use_bias, audio_features_path)

        self.out_dimension = out_dimension if out_dimension is not None else embedding_dim

        # input_dimをembedding_dimに設定
        input_dim = self.embedding_dim  
        self.linear_layer = nn.Linear(input_dim, self.out_dimension, bias=self.use_bias)

        # Make embedding_layer accessible
        self.embedding_layer = self.embedding_ext.embedding_layer

        print(f'Built EmbeddingWWithAudioFeatures model\n'
              f'- input_dim: {input_dim}\n'
              f'- out_dimension: {self.out_dimension}\n'
              f'- use_bias: {self.use_bias}')

    def forward(self, o_idxs: torch.Tensor, audio_features: torch.Tensor = None) -> torch.Tensor:
        embeddings = super().forward(o_idxs, audio_features)

        # デバッグ: embeddingsの形状を確認
        # print(f"Embeddings shape before linear layer: {embeddings.shape}")
        # print(f"Linear layer weight shape: {self.linear_layer.weight.shape}")

        # 線形層適用
        combined_embeddings = self.linear_layer(embeddings)

        # デバッグ: 出力の形状を出力
        # print(f"Output shape after linear layer: {combined_embeddings.shape}")

        return combined_embeddings



class AnchorBasedCollaborativeFiltering(FeatureExtractor):
    """
    Anchor-based Collaborative Filtering by Barkan et al. (https://dl.acm.org/doi/10.1145/3459637.3482056) published at CIKM 2021.
    """

    def __init__(self, n_objects: int, embedding_dim: int, anchors: nn.Parameter, delta_exc: float = 0,
                 delta_inc: float = 0, max_norm: float = None):
        super().__init__()
        """
        NB. delta_inc and delta_exc should be passed only when instantiating this FeatureExtractor for Items.

        :param n_objects: number of objects in the system (users or items)
        :param embedding_dim: embedding dimension
        :param anchors: nn.Parameters with shape (n_anchors,embedding_dim)
        :param delta_exc: factor multiplied to the exclusiveness loss
        :param delta_inc: factor multiplied to the inclusiveness loss
        :param max_norm: max norm of the l2 norm of the embeddings. 
        """

        self.anchors = anchors
        self.n_anchors = anchors.shape[0]
        self.delta_exc = delta_exc
        self.delta_inc = delta_inc

        self.embedding_ext = Embedding(n_objects, embedding_dim, max_norm)

        self._acc_exc = 0.
        self._acc_inc = 0.
        self.name = "AnchorBasedCollaborativeFiltering"

        print(f'Built AnchorBasedCollaborativeFiltering module \n'
              f'- n_anchors: {self.n_anchors} \n'
              f'- delta_exc: {self.delta_exc} \n'
              f'- delta_inc: {self.delta_inc} \n')

    def init_parameters(self):
        torch.nn.init.normal_(self.anchors, 0, 1)
        torch.nn.init.normal_(self.embedding_ext.embedding_layer.weight, 0, 1)  # Overriding previous init

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        """
        :param o_idxs: Shape is either [batch_size] or [batch_size,n_neg_p_1]
        :return:
        """
        assert o_idxs is not None, "Object indexes not provided"
        assert len(o_idxs.shape) == 2 or len(o_idxs.shape) == 1, \
            f'Object indexes have shape that does not match the network ({o_idxs.shape})'

        o_embed = self.embedding_ext(o_idxs)  # [...,embedding_dim]

        o_dots = o_embed @ self.anchors.T  # [...,n_anchors]

        o_coeff = nn.Softmax(dim=-1)(o_dots)  # [...,n_anchors]

        o_vect = o_coeff @ self.anchors  # [...,embedding_dim]

        # Exclusiveness constraint (BCE)
        exc = - (o_coeff * torch.log(o_coeff)).sum()

        # Inclusiveness constraint
        q_k = o_coeff.reshape(-1, self.n_anchors).sum(axis=0).div(o_coeff.sum())  # [n_anchors]
        inc = - (q_k * torch.log(q_k)).sum()

        self._acc_exc += exc
        self._acc_inc += inc

        return o_vect

    def get_and_reset_loss(self) -> float:
        acc_inc, acc_exc = self._acc_inc, self._acc_exc
        self._acc_inc = self._acc_exc = 0
        return - self.delta_inc * acc_inc + self.delta_exc * acc_exc


class PrototypeEmbedding(FeatureExtractor):
    """
    ProtoMF building block. It represents an object (item/user) given the similarity with the prototypes.
    """

    def __init__(self, n_objects: int, embedding_dim: int, n_prototypes: int = None, use_weight_matrix: bool = False,
                 sim_proto_weight: float = 1., sim_batch_weight: float = 1.,
                 reg_proto_type: str = 'soft', reg_batch_type: str = 'soft', cosine_type: str = 'shifted',
                 max_norm: float = None):
        """
        :param n_objects: number of objects in the system (users or items)
        :param embedding_dim: embedding dimension
        :param n_prototypes: number of prototypes to consider. If none, is set to be embedding_dim.
        :param use_weight_matrix: Whether to use a linear layer after the prototype layer.
        :param sim_proto_weight: factor multiplied to the regularization loss for prototypes
        :param sim_batch_weight: factor multiplied to the regularization loss for batch
        :param reg_proto_type: type of regularization applied batch-prototype similarity matrix on the prototypes. Possible values are ['max','soft','incl']
        :param reg_batch_type: type of regularization applied batch-prototype similarity matrix on the batch. Possible values are ['max','soft']
        :param cosine_type: type of cosine similarity to apply. Possible values ['shifted','standard','shifted_and_div']
        :param max_norm: max norm of the l2 norm of the embeddings.

        """

        super(PrototypeEmbedding, self).__init__()

        self.n_objects = n_objects
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes
        self.use_weight_matrix = use_weight_matrix
        self.sim_proto_weight = sim_proto_weight
        self.sim_batch_weight = sim_batch_weight
        self.reg_proto_type = reg_proto_type
        self.reg_batch_type = reg_batch_type
        self.cosine_type = cosine_type

        self.embedding_ext = Embedding(n_objects, embedding_dim, max_norm)

        if self.n_prototypes is None:
            self.prototypes = nn.Parameter(torch.randn([self.embedding_dim, self.embedding_dim]))
            self.n_prototypes = self.embedding_dim
        else:
            self.prototypes = nn.Parameter(torch.randn([self.n_prototypes, self.embedding_dim]))

        if self.use_weight_matrix:
            self.weight_matrix = nn.Linear(self.n_prototypes, self.embedding_dim, bias=False)

        # Cosine Type
        if self.cosine_type == 'standard':
            self.cosine_sim_func = nn.CosineSimilarity(dim=-1)
        elif self.cosine_type == 'shifted':
            self.cosine_sim_func = lambda x, y: (1 + nn.CosineSimilarity(dim=-1)(x, y))
        elif self.cosine_type == 'shifted_and_div':
            self.cosine_sim_func = lambda x, y: (1 + nn.CosineSimilarity(dim=-1)(x, y)) / 2
        else:
            raise ValueError(f'Cosine type {self.cosine_type} not implemented')

        # Regularization Batch
        if self.reg_batch_type == 'max':
            self.reg_batch_func = lambda x: - x.max(dim=1).values.mean()
        elif self.reg_batch_type == 'soft':
            self.reg_batch_func = lambda x: self._entropy_reg_loss(x, 1)
        else:
            raise ValueError(f'Regularization Type for Batch {self.reg_batch_func} not yet implemented')

        # Regularization Proto
        if self.reg_proto_type == 'max':
            self.reg_proto_func = lambda x: - x.max(dim=0).values.mean()
        elif self.reg_proto_type == 'soft':
            self.reg_proto_func = lambda x: self._entropy_reg_loss(x, 0)
        elif self.reg_proto_type == 'incl':
            self.reg_proto_func = lambda x: self._inclusiveness_constraint(x)
        else:
            raise ValueError(f'Regularization Type for Proto {self.reg_proto_type} not yet implemented')

        self._acc_r_proto = 0
        self._acc_r_batch = 0
        self.name = "PrototypeEmbedding"

        print(f'Built PrototypeEmbedding model \n'
              f'- n_prototypes: {self.n_prototypes} \n'
              f'- use_weight_matrix: {self.use_weight_matrix} \n'
              f'- sim_proto_weight: {self.sim_proto_weight} \n'
              f'- sim_batch_weight: {self.sim_batch_weight} \n'
              f'- reg_proto_type: {self.reg_proto_type} \n'
              f'- reg_batch_type: {self.reg_batch_type} \n'
              f'- cosine_type: {self.cosine_type} \n')

    @staticmethod
    def _entropy_reg_loss(sim_mtx, axis: int):
        o_coeff = nn.Softmax(dim=axis)(sim_mtx)
        entropy = - (o_coeff * torch.log(o_coeff)).sum(axis=axis).mean()
        return entropy

    @staticmethod
    def _inclusiveness_constraint(sim_mtx):
        '''
        NB. This method is applied only on a square matrix (batch_size,n_prototypes) and it return the negated
        inclusiveness constraints (its minimization brings more equal load sharing among the prototypes)
        '''
        o_coeff = nn.Softmax(dim=1)(sim_mtx)
        q_k = o_coeff.sum(axis=0).div(o_coeff.sum())  # [n_prototypes]
        entropy_q_k = - (q_k * torch.log(q_k)).sum()
        return - entropy_q_k

    def init_parameters(self):
        if self.use_weight_matrix:
            nn.init.xavier_normal_(self.weight_matrix.weight)

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        """
        :param o_idxs: Shape is either [batch_size] or [batch_size,n_neg_p_1]
        :return:
        """
        assert o_idxs is not None, "Object indexes not provided"
        assert len(o_idxs.shape) == 2 or len(o_idxs.shape) == 1, \
            f'Object indexes have shape that does not match the network ({o_idxs.shape})'

        o_embed = self.embedding_ext(o_idxs)  # [..., embedding_dim]

        # https://github.com/pytorch/pytorch/issues/48306
        sim_mtx = self.cosine_sim_func(o_embed.unsqueeze(-2), self.prototypes)  # [..., n_prototypes]

        if self.use_weight_matrix:
            w = self.weight_matrix(sim_mtx)  # [...,embedding_dim]
        else:
            w = sim_mtx  # [..., embedding_dim = n_prototypes]

        # Computing additional losses
        batch_proto = sim_mtx.reshape([-1, sim_mtx.shape[-1]])

        self._acc_r_batch += self.reg_batch_func(batch_proto)
        self._acc_r_proto += self.reg_proto_func(batch_proto)

        return w

    def get_and_reset_loss(self) -> float:
        acc_r_proto, acc_r_batch = self._acc_r_proto, self._acc_r_batch
        self._acc_r_proto = self._acc_r_batch = 0
        return self.sim_proto_weight * acc_r_proto + self.sim_batch_weight * acc_r_batch

# 音源特徴を用いたProtoMFモデル
class PrototypeEmbeddingWithAudio(PrototypeEmbedding):
    """
    PrototypeEmbedding に音源特徴量を追加したモデル。
    """

    def __init__(self, n_objects: int, embedding_dim: int, n_prototypes: int = None, 
                 audio_feature_dim: int = 13, use_audio_bias: bool = False,
                 audio_feature_path: str = None, **kwargs):
        """
        :param n_objects: オブジェクト（ユーザーやアイテム）の数
        :param embedding_dim: 埋め込み次元
        :param n_prototypes: プロトタイプの数
        :param audio_feature_dim: 音源特徴量の次元
        :param use_audio_bias: 音源特徴量処理にバイアスを追加するかどうか
        :param audio_features_path: 音源特徴量データのパス（ロード時に使用）
        :param kwargs: PrototypeEmbedding の他の引数
        """
        super().__init__(n_objects=n_objects, embedding_dim=embedding_dim, n_prototypes=n_prototypes, **kwargs)

        self.audio_feature_dim = audio_feature_dim
        self.use_audio_bias = use_audio_bias
        self.audio_feature_path = audio_feature_path

        # 音源特徴量用の線形レイヤ
        self.audio_feature_layer = nn.Linear(audio_feature_dim, embedding_dim, bias=use_audio_bias)

        print(f'PrototypeEmbeddingWithAudioFeatures initialized \n'
              f'- audio_feature_dim: {self.audio_feature_dim} \n'
              f'- use_audio_bias: {self.use_audio_bias} \n'
              f'- audio_feature_path: {self.audio_feature_path}')

    def forward(self, o_idxs: torch.Tensor, audio_features: torch.Tensor = None) -> torch.Tensor:
        """
        :param o_idxs: オブジェクトインデックス
        :param audio_features: 音源特徴量
        :return: 埋め込みベクトル
        """
        # 元の埋め込みを取得
        embeddings = super().forward(o_idxs)  # PrototypeEmbedding の forward を呼び出し

        # 音源特徴量が提供されている場合、それを埋め込みに加算
        if audio_features is not None:
            audio_embeddings = self.audio_feature_layer(audio_features)
            embeddings += audio_embeddings  # 音源特徴量を埋め込みに加算

        return embeddings

    def _get_audio_features(self, o_idxs: torch.Tensor) -> torch.Tensor:
        """
        オブジェクトインデックスに対応する音源特徴量を取得。
        :param o_idxs: オブジェクトインデックス
        :return: 音源特徴量テンソル
        """
        try:
            item_ids = o_idxs.cpu().numpy().flatten()  # インデックスをフラット化
            audio_features = self.audio_features.iloc[item_ids].loc[:, 'feature_1':'feature_13'].values
            
            # Tensor に変換し、デバイスを揃える
            audio_features_tensor = torch.tensor(audio_features, dtype=torch.float32).to(o_idxs.device)
            
            return audio_features_tensor
        
        except Exception as e:
            print(f"Error fetching audio features for indices {o_idxs}: {e}")
            raise



class ConcatenateFeatureExtractors(FeatureExtractor):

    def __init__(self, model_1: FeatureExtractor, model_2: FeatureExtractor, invert: bool = False):
        super().__init__()

        """
        Concatenates the latent dimension (considered in position -1) of two Feature Extractors models.
        :param model_1: a FeatureExtractor model
        :param model_2: a FeatureExtractor model
        :param invert: whether to place the latent representation from the second model on top.
        """

        self.model_1 = model_1
        self.model_2 = model_2
        self.invert = invert

        self.name = 'ConcatenateFeatureExtractors'
        
        # Debug log to confirm the models being used
        print(f"model_1 type: {type(self.model_1)} | model_1 name: {self.model_1.name}")
        print(f"model_2 type: {type(self.model_2)} | model_2 name: {self.model_2.name}")

        print(f'Built ConcatenateFeatureExtractors model \n'
              f'- model_1: {self.model_1.name} \n'
              f'- model_2: {self.model_2.name} \n'
              f'- invert: {self.invert} \n')

    def forward(self, o_idxs: torch.Tensor) -> torch.Tensor:
        o_repr_1 = self.model_1(o_idxs)
        o_repr_2 = self.model_2(o_idxs)

        if self.invert:
            return torch.cat([o_repr_2, o_repr_1], dim=-1)
        else:
            return torch.cat([o_repr_1, o_repr_2], dim=-1)

    def get_and_reset_loss(self) -> float:
        loss_1 = self.model_1.get_and_reset_loss()
        loss_2 = self.model_2.get_and_reset_loss()
        return loss_1 + loss_2

    def init_parameters(self):
        self.model_1.init_parameters()
        self.model_2.init_parameters()
