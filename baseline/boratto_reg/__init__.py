"""Boratto-reg baseline package."""

__all__ = ["pairwise_boratto_reg_prior_attention"]


def __getattr__(name):
    if name == "pairwise_boratto_reg_prior_attention":
        from .loss import pairwise_boratto_reg_prior_attention

        return pairwise_boratto_reg_prior_attention
    raise AttributeError(name)
