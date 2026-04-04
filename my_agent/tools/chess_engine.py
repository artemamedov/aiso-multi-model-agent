import chess
import chess.engine
import shutil
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image


# ── CNN-based board reader (board_to_fen compatible, Keras 3) ────────────

_CATEGORIES = [
    "bishop_black", "bishop_white", "empty", "king_black", "king_white",
    "knight_black", "knight_white", "pawn_black", "pawn_white",
    "queen_black", "queen_white", "rook_black", "rook_white",
]

_LEGEND = {
    "pawn_white": "P", "pawn_black": "p",
    "knight_white": "N", "knight_black": "n",
    "bishop_white": "B", "bishop_black": "b",
    "rook_white": "R", "rook_black": "r",
    "queen_white": "Q", "queen_black": "q",
    "king_white": "K", "king_black": "k",
    "empty": "1",
}

_board_model = None


def _load_board_model():
    """Build CNN and load board_to_fen weights (Keras 3 compatible)."""
    global _board_model
    if _board_model is not None:
        return _board_model

    import importlib.resources as pkg_resources
    import keras

    import board_to_fen.saved_models as saved_models

    model = keras.Sequential([
        keras.layers.Input(shape=(50, 50, 3)),
        keras.layers.Conv2D(50, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(100, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(100, (3, 3), activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(13, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    weights_path = str(pkg_resources.files(saved_models) / "november_model_weights.h5")
    model.load_weights(weights_path)
    _board_model = model
    return model


def _image_to_fen(image_path: str, black_view: bool = False) -> str:
    """Read a chess board image and return FEN using CNN classification."""
    model = _load_board_model()
    img = Image.open(image_path).resize((400, 400), Image.NEAREST).convert("RGB")

    # Split into 64 tiles (50x50 each)
    tiles = []
    for row in range(8):
        for col in range(8):
            box = (col * 50, row * 50, (col + 1) * 50, (row + 1) * 50)
            tiles.append(img.crop(box))

    # Classify each tile
    squares = []
    for tile in tiles:
        arr = np.array(tile).reshape(1, 50, 50, 3)
        pred = model.predict(arr, verbose=0)
        squares.append(_CATEGORIES[np.argmax(pred)])

    # Convert to FEN
    long_fen = ""
    for i, square in enumerate(squares):
        if i % 8 == 0 and i > 0:
            long_fen += "/"
        long_fen += _LEGEND[square]

    fen = long_fen
    for n in range(8, 0, -1):
        fen = fen.replace("1" * n, str(n))

    if black_view:
        fen = fen[::-1]
    return fen


# ── Public tool functions ────────────────────────────────────────────────


def analyze_chess_position(fen: str) -> str:
    """Analyze a chess position using Stockfish and return the best move.

    Use this when you already have a FEN string for a chess position.

    Args:
        fen: FEN string describing the chess position.
             Example: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

    Returns:
        The best move in standard algebraic notation (e.g., "Rd5", "Nf3").
    """
    time_limit = 5.0
    stockfish_path = shutil.which("stockfish")
    if not stockfish_path:
        return "Error: Stockfish not found. Install with: brew install stockfish"

    try:
        board = chess.Board(fen)
    except ValueError as e:
        return f"Error: Invalid FEN string: {e}"

    if not board.is_valid():
        return f"Error: Invalid board position from FEN: {fen}"

    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            result = engine.analyse(board, chess.engine.Limit(time=time_limit))
            best_move = result.get("pv", [None])[0]
            if best_move is None:
                return "Error: Stockfish could not find a move."
            # Convert to SAN (standard algebraic notation like Rd5, Nf3)
            san = board.san(best_move)
            return san
    except Exception as e:
        return f"Error running Stockfish: {e}"


def read_chess_board(image_path: str, active_color: str) -> str:
    """Read a chess board image and return the best move using Stockfish.

    This tool uses computer vision (CNN) to recognize pieces on the board,
    then analyzes the position with Stockfish. Use this instead of trying
    to read the board yourself.

    Args:
        image_path: Path to the chess board image file.
        active_color: Whose turn it is. Must be "w" for white or "b" for black.

    Returns:
        The best move in standard algebraic notation (e.g., "Rd5", "Nf3"),
        or an error message if the board cannot be read.
    """
    path = Path(image_path)
    if not path.exists():
        return f"Error: Image file not found: {image_path}"

    # Try both orientations and pick the one where kings are on natural ranks
    candidates = []
    for black_view in [False, True]:
        try:
            fen_board = _image_to_fen(image_path, black_view=black_view)
            full_fen = f"{fen_board} {active_color} - - 0 1"
            board = chess.Board(full_fen)
            if not board.is_valid():
                continue

            # Score orientation: kings on natural ranks are more likely correct
            # White king (K) should be on ranks 1-3, Black king (k) on ranks 6-8
            score = 0
            wk = board.king(chess.WHITE)
            bk = board.king(chess.BLACK)
            if wk is not None:
                wk_rank = chess.square_rank(wk)  # 0-based (0=rank1)
                score += 3 if wk_rank <= 2 else 0
            if bk is not None:
                bk_rank = chess.square_rank(bk)
                score += 3 if bk_rank >= 5 else 0

            move = analyze_chess_position(full_fen)
            if move.startswith("Error"):
                continue

            candidates.append((score, move))
        except Exception:
            continue

    if candidates:
        # Pick the orientation with highest score (kings on natural ranks)
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    return "Error: Could not read a valid chess position from the image."
