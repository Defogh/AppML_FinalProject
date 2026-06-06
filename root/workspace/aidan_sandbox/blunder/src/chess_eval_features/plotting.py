"""Visual diagnostics for extracted chess features."""

import re

import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PLOT_STYLE = {
  "white": "tab:blue",
  "black": "tab:orange",
  "opening": "tab:green",
  "middlegame": "tab:purple",
  "endgame": "tab:brown",
  "white_castle": "tab:blue",
  "black_castle": "tab:orange",
  "white_queen_lost": "navy",
  "black_queen_lost": "darkorange",
  "mate_eval": "tab:red",
}

ANNOTATION_COLORS = {
  "!!": "deepskyblue",
  "!": "limegreen",
  "!?": "turquoise",
  "?!": "gold",
  "?": "darkorange",
  "??": "darkred",
}

ANNOTATION_PATTERN = re.compile(r"([!?]+)$")


def get_annotation_color(annotation):
  return ANNOTATION_COLORS.get(annotation, "black")


def extract_move_annotation(san_raw):
  match = ANNOTATION_PATTERN.search(str(san_raw))

  if match is None:
    return None

  return match.group(1)


def is_castle_move(san_raw):
  san = str(san_raw)
  return san.startswith("O-O") or san.startswith("0-0")


def get_game_index_from_position(df_plies_feat, game_pos):
  game_indices = (
    df_plies_feat["game_index"]
    .drop_duplicates()
    .sort_values()
    .to_list()
  )

  return game_indices[game_pos]


def prepare_game_plot_data(df_plies_feat, game_pos=0):
  game_index = get_game_index_from_position(df_plies_feat, game_pos)

  df_game = (
    df_plies_feat[df_plies_feat["game_index"] == game_index]
    .copy()
    .sort_values("ply")
    .reset_index(drop=True)
  )

  df_game["white_eval"] = df_game["eval_proxy"]
  df_game["black_eval"] = -df_game["eval_proxy"]
  df_game["move_annotation"] = (
    df_game["san_raw"].apply(extract_move_annotation)
  )
  df_game["is_annotated_move"] = df_game["move_annotation"].notna()
  df_game["is_castle"] = df_game["san_raw"].apply(is_castle_move)
  df_game["white_castled"] = (
    df_game["is_castle"] & (df_game["side"] == "white")
  )
  df_game["black_castled"] = (
    df_game["is_castle"] & (df_game["side"] == "black")
  )
  df_game["white_queen_lost"] = df_game["white_queen_count"].diff() < 0
  df_game["black_queen_lost"] = df_game["black_queen_count"].diff() < 0
  df_game["queen_lost"] = (
    df_game["white_queen_lost"] | df_game["black_queen_lost"]
  )

  return game_index, df_game


def choose_plot_layout(df_game):
  n_plies = len(df_game)
  n_annotated = int(df_game["is_annotated_move"].sum())

  width = 13 + 0.045 * n_plies + 0.025 * n_annotated
  width = min(max(width, 14), 24)

  height = 10.5 + 0.012 * n_annotated
  height = min(max(height, 10.5), 15)

  if n_annotated <= 20:
    label_mode = "all"
    annotation_fontsize = 11
  elif n_annotated <= 60:
    label_mode = "extreme"
    annotation_fontsize = 10
  else:
    label_mode = "none"
    annotation_fontsize = 10

  return {
    "figsize": (width, height),
    "label_mode": label_mode,
    "annotation_fontsize": annotation_fontsize,
  }


def make_game_title(game_index, df_games=None):
  if df_games is None or "game_index" not in df_games.columns:
    return f"game_index={game_index}"

  row = df_games[df_games["game_index"] == game_index]

  if len(row) == 0:
    return f"game_index={game_index}"

  row = row.iloc[0]
  white = row.get("white", "White")
  black = row.get("black", "Black")
  result = row.get("result", "?")
  white_elo = row.get("white_elo", "?")
  black_elo = row.get("black_elo", "?")
  opening = row.get("opening", "")

  title = (
    f"{white} ({white_elo}) vs {black} ({black_elo}), "
    f"result={result}, game_index={game_index}"
  )

  if isinstance(opening, str) and len(opening) > 0:
    title += f"\n{opening}"

  return title


def should_label_annotation(annotation, label_mode):
  if annotation is None:
    return False

  if label_mode == "none":
    return False

  if label_mode == "extreme":
    return annotation in {"??", "!!"}

  if label_mode == "important":
    return annotation in {"??", "!!", "?!", "!?"}

  if label_mode == "all":
    return True

  return False


def add_vertical_event_lines(ax, df_game):
  for _, row in df_game[df_game["white_castled"]].iterrows():
    ax.axvline(
      row["ply"],
      color=PLOT_STYLE["white_castle"],
      linestyle=":",
      linewidth=1.8,
      alpha=0.9,
      zorder=1,
    )

  for _, row in df_game[df_game["black_castled"]].iterrows():
    ax.axvline(
      row["ply"],
      color=PLOT_STYLE["black_castle"],
      linestyle=":",
      linewidth=1.8,
      alpha=0.9,
      zorder=1,
    )

  for _, row in df_game[df_game["white_queen_lost"]].iterrows():
    ax.axvline(
      row["ply"],
      color=PLOT_STYLE["white_queen_lost"],
      linestyle="--",
      linewidth=2.4,
      alpha=0.85,
      zorder=1,
    )

  for _, row in df_game[df_game["black_queen_lost"]].iterrows():
    ax.axvline(
      row["ply"],
      color=PLOT_STYLE["black_queen_lost"],
      linestyle="--",
      linewidth=2.4,
      alpha=0.85,
      zorder=1,
    )

  for _, row in df_game[df_game["is_mate_eval"]].iterrows():
    ax.axvline(
      row["ply"],
      color=PLOT_STYLE["mate_eval"],
      linestyle="-.",
      linewidth=1.8,
      alpha=0.8,
      zorder=1,
    )


def add_move_annotations(
  ax,
  df_game,
  white_y_col,
  black_y_col,
  label_mode="extreme",
  annotation_fontsize=10,
):
  annotated = df_game[df_game["is_annotated_move"]].copy()

  for i, (_, row) in enumerate(annotated.iterrows()):
    x = row["ply"]
    label = row["move_annotation"]
    side = row["side"]

    if side == "white":
      y_base = row[white_y_col]
      marker = "^"
      base_offset = 14
      va = "bottom"
    else:
      y_base = row[black_y_col]
      marker = "v"
      base_offset = -14
      va = "top"

    if pd.isna(y_base):
      continue

    color = get_annotation_color(label)

    ax.scatter(
      x,
      y_base,
      s=70,
      marker=marker,
      color=color,
      edgecolor="black",
      linewidth=0.7,
      zorder=6,
    )

    if not should_label_annotation(label, label_mode):
      continue

    stagger = (i % 3) * 7
    y_offset = base_offset + stagger if side == "white" else (
      base_offset - stagger
    )

    ax.annotate(
      label,
      xy=(x, y_base),
      xytext=(0, y_offset),
      textcoords="offset points",
      fontsize=annotation_fontsize,
      fontweight="bold",
      color=color,
      ha="center",
      va=va,
      zorder=7,
      bbox={
        "boxstyle": "round,pad=0.18",
        "facecolor": "white",
        "edgecolor": color,
        "alpha": 0.9,
      },
      arrowprops={
        "arrowstyle": "-",
        "color": color,
        "alpha": 0.5,
        "linewidth": 0.6,
      },
    )


def build_common_legend_handles():
  handles = [
    Line2D(
      [0],
      [0],
      color=PLOT_STYLE["white"],
      linewidth=2.0,
      label="White eval / White material",
    ),
    Line2D(
      [0],
      [0],
      color=PLOT_STYLE["black"],
      linewidth=2.0,
      label="Black eval / Black material",
    ),
    Line2D(
      [0],
      [0],
      color=PLOT_STYLE["opening"],
      linewidth=1.8,
      linestyle="--",
      label="Opening-like weight",
    ),
    Line2D(
      [0],
      [0],
      color=PLOT_STYLE["middlegame"],
      linewidth=1.8,
      linestyle="--",
      label="Middlegame-like weight",
    ),
    Line2D(
      [0],
      [0],
      color=PLOT_STYLE["endgame"],
      linewidth=1.8,
      linestyle="--",
      label="Endgame-like weight",
    ),
    Line2D(
      [0],
      [0],
      marker="^",
      color="white",
      markerfacecolor="gray",
      markeredgecolor="black",
      linestyle="None",
      markersize=8,
      label="White annotated move",
    ),
    Line2D(
      [0],
      [0],
      marker="v",
      color="white",
      markerfacecolor="gray",
      markeredgecolor="black",
      linestyle="None",
      markersize=8,
      label="Black annotated move",
    ),
  ]

  for annotation, color in ANNOTATION_COLORS.items():
    handles.append(
      Line2D(
        [0],
        [0],
        marker="o",
        color="white",
        markerfacecolor=color,
        markeredgecolor="black",
        linestyle="None",
        markersize=8,
        label=f"{annotation} annotation",
      )
    )

  handles.extend(
    [
      Line2D(
        [0],
        [0],
        color=PLOT_STYLE["white_castle"],
        linestyle=":",
        linewidth=1.8,
        label="White castles",
      ),
      Line2D(
        [0],
        [0],
        color=PLOT_STYLE["black_castle"],
        linestyle=":",
        linewidth=1.8,
        label="Black castles",
      ),
      Line2D(
        [0],
        [0],
        color=PLOT_STYLE["white_queen_lost"],
        linestyle="--",
        linewidth=2.4,
        label="White queen lost",
      ),
      Line2D(
        [0],
        [0],
        color=PLOT_STYLE["black_queen_lost"],
        linestyle="--",
        linewidth=2.4,
        label="Black queen lost",
      ),
      Line2D(
        [0],
        [0],
        color=PLOT_STYLE["mate_eval"],
        linestyle="-.",
        linewidth=1.8,
        label="Mate eval",
      ),
      Patch(
        facecolor=PLOT_STYLE["white"],
        alpha=0.65,
        label="White loss bar",
      ),
      Patch(
        facecolor=PLOT_STYLE["black"],
        alpha=0.65,
        label="Black loss bar",
      ),
    ]
  )

  return handles


def plot_game_diagnostics(
  df_plies_feat,
  df_games=None,
  game_pos=0,
  eval_clip=20,
):
  game_index, df_game = prepare_game_plot_data(
    df_plies_feat,
    game_pos=game_pos,
  )

  layout = choose_plot_layout(df_game)
  figsize = layout["figsize"]
  label_mode = layout["label_mode"]
  annotation_fontsize = layout["annotation_fontsize"]
  title = make_game_title(game_index, df_games=df_games)

  plot_df = df_game.copy()
  plot_df["white_eval_plot"] = plot_df["white_eval"].clip(
    lower=-eval_clip,
    upper=eval_clip,
  )
  plot_df["black_eval_plot"] = plot_df["black_eval"].clip(
    lower=-eval_clip,
    upper=eval_clip,
  )

  fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=figsize,
    sharex=True,
    gridspec_kw={"height_ratios": [1.2, 1.2, 0.9]},
  )
  fig.suptitle(title, fontsize=14)

  ax_eval = axes[0]
  ax_eval_phase = ax_eval.twinx()

  ax_eval.plot(
    plot_df["ply"],
    plot_df["white_eval_plot"],
    color=PLOT_STYLE["white"],
    linewidth=2.0,
  )
  ax_eval.plot(
    plot_df["ply"],
    plot_df["black_eval_plot"],
    color=PLOT_STYLE["black"],
    linewidth=2.0,
  )
  ax_eval.axhline(0, color="black", linewidth=0.8, alpha=0.4)

  ax_eval_phase.plot(
    plot_df["ply"],
    plot_df["opening_like_weight"],
    color=PLOT_STYLE["opening"],
    linestyle="--",
    linewidth=1.8,
    alpha=0.9,
  )
  ax_eval_phase.plot(
    plot_df["ply"],
    plot_df["middlegame_like_weight"],
    color=PLOT_STYLE["middlegame"],
    linestyle="--",
    linewidth=1.8,
    alpha=0.9,
  )
  ax_eval_phase.plot(
    plot_df["ply"],
    plot_df["endgame_like_weight"],
    color=PLOT_STYLE["endgame"],
    linestyle="--",
    linewidth=1.8,
    alpha=0.9,
  )
  add_vertical_event_lines(ax_eval, plot_df)
  add_move_annotations(
    ax=ax_eval,
    df_game=plot_df,
    white_y_col="white_eval_plot",
    black_y_col="black_eval_plot",
    label_mode=label_mode,
    annotation_fontsize=annotation_fontsize,
  )
  ax_eval.set_ylabel("Eval proxy")
  ax_eval_phase.set_ylabel("Phase-like weight")
  ax_eval_phase.set_ylim(-0.05, 1.05)
  ax_eval.set_title("Eval trajectory with dynamic phase indicators")

  ax_mat = axes[1]
  ax_mat_phase = ax_mat.twinx()

  ax_mat.plot(
    plot_df["ply"],
    plot_df["white_material"],
    color=PLOT_STYLE["white"],
    linewidth=2.0,
  )
  ax_mat.plot(
    plot_df["ply"],
    plot_df["black_material"],
    color=PLOT_STYLE["black"],
    linewidth=2.0,
  )
  ax_mat_phase.plot(
    plot_df["ply"],
    plot_df["opening_like_weight"],
    color=PLOT_STYLE["opening"],
    linestyle="--",
    linewidth=1.8,
    alpha=0.9,
  )
  ax_mat_phase.plot(
    plot_df["ply"],
    plot_df["middlegame_like_weight"],
    color=PLOT_STYLE["middlegame"],
    linestyle="--",
    linewidth=1.8,
    alpha=0.9,
  )
  ax_mat_phase.plot(
    plot_df["ply"],
    plot_df["endgame_like_weight"],
    color=PLOT_STYLE["endgame"],
    linestyle="--",
    linewidth=1.8,
    alpha=0.9,
  )
  add_vertical_event_lines(ax_mat, plot_df)
  add_move_annotations(
    ax=ax_mat,
    df_game=plot_df,
    white_y_col="white_material",
    black_y_col="black_material",
    label_mode=label_mode,
    annotation_fontsize=annotation_fontsize,
  )
  ax_mat.set_ylabel("Material")
  ax_mat_phase.set_ylabel("Phase-like weight")
  ax_mat_phase.set_ylim(-0.05, 1.05)
  ax_mat.set_title("Material trajectory with dynamic phase indicators")

  ax_loss = axes[2]
  white_loss = plot_df["white_loss_proxy"].fillna(0)
  black_loss = plot_df["black_loss_proxy"].fillna(0)

  ax_loss.bar(
    plot_df["ply"],
    white_loss,
    color=PLOT_STYLE["white"],
    alpha=0.65,
    width=0.85,
  )
  ax_loss.bar(
    plot_df["ply"],
    -black_loss,
    color=PLOT_STYLE["black"],
    alpha=0.65,
    width=0.85,
  )
  ax_loss.axhline(0, color="black", linewidth=0.8)
  add_vertical_event_lines(ax_loss, plot_df)

  annotated = plot_df[plot_df["is_annotated_move"]]

  for i, (_, row) in enumerate(annotated.iterrows()):
    x = row["ply"]
    label = row["move_annotation"]
    side = row["side"]
    color = get_annotation_color(label)

    if side == "white":
      marker = "^"
      y_base = 0
      y_offset = 13 + (i % 3) * 6
      va = "bottom"
    else:
      marker = "v"
      y_base = 0
      y_offset = -13 - (i % 3) * 6
      va = "top"

    ax_loss.scatter(
      x,
      y_base,
      s=70,
      marker=marker,
      color=color,
      edgecolor="black",
      linewidth=0.7,
      zorder=6,
    )

    if not should_label_annotation(label, label_mode):
      continue

    ax_loss.annotate(
      label,
      xy=(x, y_base),
      xytext=(0, y_offset),
      textcoords="offset points",
      fontsize=annotation_fontsize,
      fontweight="bold",
      color=color,
      ha="center",
      va=va,
      zorder=7,
      bbox={
        "boxstyle": "round,pad=0.18",
        "facecolor": "white",
        "edgecolor": color,
        "alpha": 0.9,
      },
    )

  ax_loss.set_xlabel("Ply")
  ax_loss.set_ylabel("Eval loss proxy")
  ax_loss.set_title("Eval loss by side")

  legend_handles = build_common_legend_handles()
  fig.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(0.87, 0.5),
    frameon=True,
    fontsize=9,
    title="Plot details",
    title_fontsize=10,
  )

  plt.tight_layout(rect=[0, 0, 0.86, 0.96])
  plt.show()

  return plot_df
