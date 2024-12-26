from typing import Tuple

import torch
from torch import nn

# 追加
import pandas as pd

from feature_extraction.feature_extractors import FeatureExtractor, Embedding, AnchorBasedCollaborativeFiltering, \
    PrototypeEmbedding,PrototypeEmbeddingWithAudio, ConcatenateFeatureExtractors, EmbeddingW, EmbeddingWithAudioFeatures,EmbeddingWWithAudioFeatures


class FeatureExtractorFactory:

    @staticmethod
    def create_models(ft_ext_param: dict, n_users: int, n_items: int, audio_features=None) -> Tuple[FeatureExtractor, FeatureExtractor]:

        """
        Helper function to create both the user and item feature extractor. It either creates two detached
        FeatureExtractors or a single one shared by users and items.
        :param ft_ext_param: parameters for the user feature extractor model. ft_ext_param.ft_type is used for
            switching between models.
        :param n_users: number of users in the system.
        :param n_items: number of items in the system.
        :return: [user_feature_extractor, item_feature_extractor]
        """
        assert 'ft_type' in ft_ext_param, "Type has not been specified for FeatureExtractor! " \
                                          "FeatureExtractor model not created"
        ft_type = ft_ext_param['ft_type']
        embedding_dim = ft_ext_param['embedding_dim']



        if ft_type == 'detached':
            # Build the extractors independently (e.g. two embeddings branches, one for users and one for items)
            print("🚀user_feature_extractorを作成します") 
            user_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'], n_users,
                                                                          embedding_dim)
            print("🚀item_feature_extractorを作成します")                                
            item_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['item_ft_ext_param'], n_items,
                                                                          embedding_dim)
            return user_feature_extractor, item_feature_extractor

        elif ft_type == 'prototypes':
            # The feature extractors are related, e.g. one of them contains a prototype layer and the other an embedding
            if 'prototypes' in ft_ext_param['user_ft_ext_param']['ft_type'] and \
                    ft_ext_param['item_ft_ext_param']['ft_type'] == 'embedding':
                # User Proto

                user_n_prototypes = ft_ext_param['user_ft_ext_param']['n_prototypes']

                print("🚀user_feature_extractorを作成します")
                user_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'],
                                                                              n_users,embedding_dim)
                print("🚀item_feature_extractorを作成します")                                                              
                item_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['item_ft_ext_param'],
                                                                              n_items,
                                                                              user_n_prototypes)

            elif 'prototypes' in ft_ext_param['item_ft_ext_param']['ft_type'] and \
                    ft_ext_param['user_ft_ext_param']['ft_type'] == 'embedding':
                # Item Proto
                item_n_prototypes = ft_ext_param['item_ft_ext_param']['n_prototypes']

                print("🚀user_feature_extractorを作成します") 
                user_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'],
                                                                              n_users,item_n_prototypes)
                print("🚀item_feature_extractorを作成します")                                                              
                item_feature_extractor = FeatureExtractorFactory.create_model(ft_ext_param['item_ft_ext_param'],
                                                                              n_items,
                                                                              embedding_dim)

            else:
                raise ValueError('Combination of ft_type of user/item feature extractors not valid for prototypes')

            return user_feature_extractor, item_feature_extractor

        elif ft_type == 'prototypes_double_tie':
            # User-Item Proto
            item_n_prototypes = ft_ext_param['item_ft_ext_param']['n_prototypes']
            user_n_prototypes = ft_ext_param['user_ft_ext_param']['n_prototypes']
            user_use_weight_matrix = ft_ext_param['user_ft_ext_param']['use_weight_matrix']
            item_use_weight_matrix = ft_ext_param['item_ft_ext_param']['use_weight_matrix']

            assert not user_use_weight_matrix and not item_use_weight_matrix, 'Use Weight Matrix should be turned off to tie the weights!'

            # Building User Proto branch
            ft_ext_param['user_ft_ext_param']['ft_type'] = 'prototypes'
            print("🚀user_protoを作成します") 
            user_proto = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'], n_users, embedding_dim)
            
            ft_ext_param['user_ft_ext_param']['ft_type'] = 'embedding_w'
            ft_ext_param['user_ft_ext_param']['out_dimension'] = item_n_prototypes
            print("🚀user_embedを作成します") 
            user_embed = FeatureExtractorFactory.create_model(ft_ext_param['user_ft_ext_param'], n_users, embedding_dim)

            # Building Item Proto branch
            ft_ext_param['item_ft_ext_param']['ft_type'] = 'prototypes'
            print("🚀item_protoを作成します")
            item_proto = FeatureExtractorFactory.create_model(ft_ext_param['item_ft_ext_param'], n_items, embedding_dim)
            
            ft_ext_param['item_ft_ext_param']['ft_type'] = 'embedding_w'
            ft_ext_param['item_ft_ext_param']['out_dimension'] = user_n_prototypes
            print("🚀item_embedを作成します")
            item_embed = FeatureExtractorFactory.create_model(ft_ext_param['item_ft_ext_param'], n_items, embedding_dim)

            # Tying the weights together
            user_embed.embedding_layer.weight = user_proto.embedding_ext.embedding_layer.weight
            item_embed.embedding_layer.weight = item_proto.embedding_ext.embedding_layer.weight
            
            # 特徴抽出器の作成
            print("🚀user_feature_extractorを作成します")
            user_feature_extractor = ConcatenateFeatureExtractors(user_proto, user_embed, invert=False)
            print("🚀item_feature_extractorを作成します")
            item_feature_extractor = ConcatenateFeatureExtractors(item_proto, item_embed, invert=True)

            return user_feature_extractor, item_feature_extractor

        elif ft_type == 'acf':
            # Anchor-based collaborative filtering

            n_anchors = ft_ext_param['n_anchors']
            delta_exc = ft_ext_param['delta_exc']
            delta_inc = ft_ext_param['delta_inc']
            max_norm = ft_ext_param['max_norm'] if 'max_norm' in ft_ext_param else None
            # Create shared parameters
            anchors = nn.Parameter(torch.randn(n_anchors, embedding_dim))

            user_feature_extractor = AnchorBasedCollaborativeFiltering(n_users, embedding_dim, anchors,
                                                                       max_norm=max_norm)
            item_feature_extractor = AnchorBasedCollaborativeFiltering(n_items, embedding_dim, anchors, delta_exc,
                                                                       delta_inc, max_norm=max_norm)
            return user_feature_extractor, item_feature_extractor
        else:
            raise ValueError(f'FeatureExtractor <{ft_type}> Not Implemented..yet')

    @staticmethod
    def create_model(ft_ext_param: dict, n_objects: int, embedding_dim: int) -> FeatureExtractor:
        """
        Creates the specified FeatureExtractor model by reading the ft_ext_param. Currently available:
        - Embedding: represents objects by learning an embedding, A.K.A. Collaborative Filtering.
        - EmbeddingW: As Embedding but followed by a linear layer.
        - PrototypeEmbedding: represents an object by the similarity to the prototypes.

        :param ft_ext_param: parameters specific for the model type. ft_ext_param.ft_type is used for switching between
                models.
        :param embedding_dim: dimension of the final embeddings
        :param n_objects: number of objects in the system
        """

        ft_type = ft_ext_param["ft_type"]

        print('--- Building FeatureExtractor model ---')
        if ft_type == 'embedding':
            max_norm = ft_ext_param['max_norm'] if 'max_norm' in ft_ext_param else None
            only_positive = ft_ext_param['only_positive'] if 'only_positive' in ft_ext_param else False
            # バグったらこれをコメントインして、A以下をコメントアウトして下さい
            # model = Embedding(n_objects=n_objects, embedding_dim=embedding_dim)

            # Aここから
            # 音源特徴を含むデータパス
            audio_feature_path = "/content/drive/MyDrive/Master/research/ProtoMF/data/lfm2b-1mon/data_with_CLMR/train_data_with_features.csv"

            # 音源特徴量のCSVファイルパスと音源特徴量の次元を設定
            audio_feature_path = ft_ext_param.get('audio_features_path', None)  # パスを指定
            audio_feature_dim = ft_ext_param.get('audio_feature_dim', 13)  # デフォルトで13次元（音源特徴量の次元数）
            
            
            print("ft_ext_param:", ft_ext_param)

            if audio_feature_path is not None and audio_feature_path.strip() != "":
                # 新しい埋め込み層（音源特徴量を統合したもの）を作成
                print("音源特徴を含めて実行します")
                model = EmbeddingWithAudioFeatures(
                    n_objects=n_objects, 
                    embedding_dim=embedding_dim, 
                    audio_feature_dim=audio_feature_dim,
                    audio_features_path=audio_feature_path
                )
            else:
                # もし音源特徴量が無ければ、従来通りの埋め込み層を使用
                print("音源特徴なしで実行します")
                model = Embedding(n_objects=n_objects, embedding_dim=embedding_dim)
            
            print(f"- audio_feature_path: {audio_feature_path}")  # パスの状態を出力

            # Aここまで
        
        elif ft_type == 'embedding_w':
            max_norm = ft_ext_param['max_norm'] if 'max_norm' in ft_ext_param else None
            out_dimension = ft_ext_param['out_dimension'] if 'out_dimension' in ft_ext_param else None
            use_bias = ft_ext_param['use_bias'] if 'use_bias' in ft_ext_param else False

            # 音源特徴量のCSVファイルパスと音源特徴量の次元を設定
            audio_feature_path = ft_ext_param.get('audio_features_path', None)  # パスを指定
            audio_feature_dim = ft_ext_param.get('audio_feature_dim', 13)  # デフォルトで13次元（音源特徴量の次元数）

            print("ft_ext_param:", ft_ext_param)

            if audio_feature_path is not None and audio_feature_path.strip() != "":
                # 新しい埋め込み層（音源特徴量を統合したもの）を作成
                print("音源特徴を含めて実行します")
                print("Creating EmbeddingWWithAudioFeatures with parameters:")
                print(f"n_objects: {n_objects}, embedding_dim: {embedding_dim}, "
                      f"audio_feature_dim: {audio_feature_dim}, max_norm: {max_norm}, "
                      f"out_dimension: {out_dimension}, use_bias: {use_bias}, "
                      f"audio_features_path: {audio_feature_path}")

                model = EmbeddingWWithAudioFeatures(
                    n_objects, 
                    embedding_dim,
                    audio_feature_dim,  # この引数を一度だけ渡します
                    max_norm=max_norm,  # 引数名を明示的に指定
                    out_dimension=out_dimension,
                    use_bias=use_bias,
                    audio_features_path=audio_feature_path
                )
            else:
                # もし音源特徴量が無ければ、従来通りの埋め込み層を使用
                print("音源特徴なしで実行します")
                model = EmbeddingW(n_objects, embedding_dim, max_norm, out_dimension, use_bias)

            print(f"- audio_feature_path: {audio_feature_path}")  # パスの状態を出力


        elif ft_type == 'prototypes':
            n_prototypes = ft_ext_param['n_prototypes'] if 'n_prototypes' in ft_ext_param else None
            max_norm = ft_ext_param['max_norm'] if 'max_norm' in ft_ext_param else None
            sim_proto_weight = ft_ext_param['sim_proto_weight'] if 'sim_proto_weight' in ft_ext_param else 1.
            sim_batch_weight = ft_ext_param['sim_batch_weight'] if 'sim_batch_weight' in ft_ext_param else 1.
            reg_proto_type = ft_ext_param['reg_proto_type'] if 'reg_proto_type' in ft_ext_param else 'soft'
            reg_batch_type = ft_ext_param['reg_batch_type'] if 'reg_batch_type' in ft_ext_param else 'soft'
            cosine_type = ft_ext_param['cosine_type'] if 'cosine_type' in ft_ext_param else 'shifted'
            use_weight_matrix = ft_ext_param['use_weight_matrix'] if 'use_weight_matrix' in ft_ext_param else False


 
            # 音源特徴量のCSVファイルパスと音源特徴量の次元を設定
            audio_feature_path = ft_ext_param.get('audio_features_path', None)  # パスを指定
            audio_feature_dim = ft_ext_param.get('audio_feature_dim', 13)  # デフォルトで13次元（音源特徴量の次元数）
            
            
            print("ft_ext_param:", ft_ext_param)

            if audio_feature_path is not None and audio_feature_path.strip() != "":
                # 新しい埋め込み層（音源特徴量を統合したもの）を作成
                print("音源特徴を含めて実行します")
                model = PrototypeEmbeddingWithAudio(
                    n_objects=n_objects,
                    embedding_dim=embedding_dim,
                    audio_feature_dim=audio_feature_dim,
                    n_prototypes=n_prototypes,
                    use_weight_matrix=use_weight_matrix, 
                    sim_proto_weight=sim_proto_weight,
                    sim_batch_weight=sim_batch_weight, 
                    reg_proto_type=reg_proto_type, 
                    reg_batch_type=reg_batch_type, 
                    cosine_type=cosine_type, 
                    max_norm=max_norm,
                    audio_feature_path=audio_feature_path
                )
            else:
                # もし音源特徴量が無ければ、従来通りの埋め込み層を使用
                print("音源特徴なしで実行します")
                model = PrototypeEmbedding(n_objects, embedding_dim, n_prototypes, use_weight_matrix, sim_proto_weight,
                                           sim_batch_weight, reg_proto_type, reg_batch_type, cosine_type, max_norm)

            print(f"- audio_feature_path: {audio_feature_path}")  # パスの状態を出力


        elif ft_type == 'acf':
            n_anchors = ft_ext_param['n_anchors']
            delta_exc = ft_ext_param['delta_exc']
            delta_inc = ft_ext_param['delta_inc']
            max_norm = ft_ext_param['max_norm'] if 'max_norm' in ft_ext_param else None

            anchors = nn.Parameter(torch.randn(n_anchors, embedding_dim))

            model = AnchorBasedCollaborativeFiltering(n_objects, embedding_dim, anchors, delta_exc, delta_inc,
                                                      max_norm=max_norm)

        else:
            raise ValueError(f'FeatureExtractor <{ft_type}> Not Implemented..yet')

        print('--- Finished building FeatureExtractor model ---\n')
        return model
