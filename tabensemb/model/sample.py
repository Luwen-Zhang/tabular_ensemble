from tabensemb.utils import *
from skopt.space import Integer, Categorical, Real
from tabensemb.model import TorchModel
from ._sample.cat_embed import CategoryEmbeddingNN
from ._sample.require_model import RequireOthersNN


class CatEmbed(TorchModel):
    def _get_program_name(self):
        return "CatEmbed"

    @staticmethod
    def _get_model_names():
        return [
            "Category Embedding",
            "Category Embedding Extend dim",
            "Require Model Autogluon LR",
            "Require Model WideDeep TabMlp",
            "Require Model WideDeep TabMlp Wrap",
            "Require Model PyTabular CatEmbed",
            "Require Model PyTabular CatEmbed Wrap",
            "Require Model Self CatEmbed",
            "Require Model ExtCatEmbed CatEmbed",
            "Require Model ExtCatEmbed CatEmbed Wrap",
        ]

    def _new_model(self, model_name, verbose, **kwargs):
        fix_kwargs = dict(
            layers=self.datamodule.args["layers"],
            datamodule=self.datamodule,
        )
        if "Category Embedding" in model_name:
            return CategoryEmbeddingNN(
                **fix_kwargs,
                embedding_dim=3,
                embed_extend_dim="Extend dim" in model_name,
                **kwargs,
            )
        elif "Require Model" in model_name:
            return RequireOthersNN(**fix_kwargs, **kwargs)

    def _space(self, model_name):
        return [
            Real(low=0.0, high=0.5, prior="uniform", name="mlp_dropout"),
            Real(low=0.0, high=0.5, prior="uniform", name="embed_dropout"),
        ] + self.trainer.SPACE

    def required_models(self, model_name: str):
        if "Require Model" in model_name:
            postfix = "_WRAP" if "Wrap" in model_name else ""
            if "Autogluon LR" in model_name:
                return ["EXTERN_AutoGluon_Linear Regression" + postfix]
            elif "WideDeep TabMlp" in model_name:
                return ["EXTERN_WideDeep_TabMlp" + postfix]
            elif "PyTabular CatEmbed" in model_name:
                return ["EXTERN_PytorchTabular_Category Embedding" + postfix]
            elif "ExtCatEmbed" in model_name:
                return ["EXTERN_ExtCatEmbed_Category Embedding" + postfix]
            elif "Self" in model_name:
                return ["Category Embedding"]
        else:
            return None

    def _initial_values(self, model_name):
        res = {
            "mlp_dropout": 0.0,
            "embed_dropout": 0.1,
        }
        res.update(self.trainer.chosen_params)
        return res

    def _conditional_validity(self, model_name: str) -> bool:
        return True

    def _prepare_custom_datamodule(self, model_name, warm_start=False):
        from tabensemb.data import DataModule

        base = self.trainer.datamodule
        if not warm_start or not hasattr(self, "datamodule"):
            datamodule = DataModule(
                config=self.trainer.datamodule.args, initialize=False
            )
            datamodule.set_data_imputer("MeanImputer")
            datamodule.set_data_derivers(
                [("UnscaledDataDeriver", {"derived_name": "Unscaled"})]
            )
            datamodule.set_data_processors(
                [("CategoricalOrdinalEncoder", {}), ("StandardScaler", {})]
            )
        else:
            datamodule = self.datamodule
        datamodule.set_data(
            base.categories_inverse_transform(base.df),
            cont_feature_names=base.cont_feature_names,
            cat_feature_names=base.cat_feature_names,
            label_name=base.label_name,
            train_indices=base.train_indices,
            val_indices=base.val_indices,
            test_indices=base.test_indices,
            verbose=False,
        )
        tmp_derived_data = base.derived_data.copy()
        tmp_derived_data.update(datamodule.derived_data)
        datamodule.derived_data = tmp_derived_data
        self.datamodule = datamodule
        return datamodule

    def _run_custom_data_module(self, df, derived_data, model_name):
        df, my_derived_data = self.datamodule.prepare_new_data(df, ignore_absence=True)
        derived_data = derived_data.copy()
        derived_data.update(my_derived_data)
        derived_data = self.datamodule.sort_derived_data(derived_data)
        return df, derived_data, self.datamodule
