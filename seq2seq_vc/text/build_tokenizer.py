from pathlib import Path
from typing import Iterable, Union

from seq2seq_vc.text.abs_tokenizer import AbsTokenizer
from seq2seq_vc.text.char_tokenizer import CharTokenizer
from seq2seq_vc.text.phoneme_tokenizer import PhonemeTokenizer
from seq2seq_vc.text.word_tokenizer import WordTokenizer


def build_tokenizer(
    token_type: str,
    non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
    remove_non_linguistic_symbols: bool = False,
    space_symbol: str = "<space>",
    delimiter: str = None,
    g2p_type: str = None,
) -> AbsTokenizer:
    """A helper function to instantiate Tokenizer"""
    if token_type == "word":
        if remove_non_linguistic_symbols and non_linguistic_symbols is not None:
            return WordTokenizer(
                delimiter=delimiter,
                non_linguistic_symbols=non_linguistic_symbols,
                remove_non_linguistic_symbols=True,
            )
        else:
            return WordTokenizer(delimiter=delimiter)

    elif token_type == "char":
        return CharTokenizer(
            non_linguistic_symbols=non_linguistic_symbols,
            space_symbol=space_symbol,
            remove_non_linguistic_symbols=remove_non_linguistic_symbols,
        )

    elif token_type == "phn":
        return PhonemeTokenizer(
            g2p_type=g2p_type,
            non_linguistic_symbols=non_linguistic_symbols,
            space_symbol=space_symbol,
            remove_non_linguistic_symbols=remove_non_linguistic_symbols,
        )

    else:
        raise ValueError(
            f"token_mode must be one of word, char or phn: " f"{token_type}"
        )
