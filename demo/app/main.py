'''
1. create a UI that can we viewed on teh web, from lab machine (use 127.0.0..?)
2. Takes input of a WHAM ,pkl file or an mp4 file (this is the leaner input)
3. runs pipeline of otkenizer+infiller, generates the pkl file for edited motion
4. displays that beside the original
5. show overlay (later step)

usage (on the lab machine):
    python demo/app/main.py --gpu 9
then from laptop:
    ssh -N -L 7860:127.0.0.1:7860 <usr>@<lab-machine>
and open http://127.0.0.1:7860
'''
import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
RUNS_DIR = APP_DIR / "runs"
GRADIO_TMP_DIR = APP_DIR / "gradio_tmp"

GRADIO_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(GRADIO_TMP_DIR))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=None, help="physical GPU id, e.g. 9 (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to expose on the lab network")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="gradio public tunnel (off by default)")
    return ap.parse_args()


def build_ui(pipeline):
    import gradio as gr

    def run(input_file, span_fraction, splice, window_only, fps, size, do_render):
        """generator so the log streams while WHAM/rendering grinds away"""
        lines = []

        def log(msg):
            print(msg, flush=True)
            lines.append(str(msg))

        if not input_file:
            yield "upload a .pkl or .mp4 first", None, None, None, None
            return

        run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        log(f"run dir: {run_dir}")
        yield "\n".join(lines), None, None, None, None

        try:
            res = pipeline.run_pipeline(
                input_path=input_file,
                out_dir=run_dir,
                span_fraction=float(span_fraction),
                splice=splice,
                fps=int(fps),
                size=int(size),
                window_only=bool(window_only),
                render=bool(do_render),
                log=log,
            )
        except Exception:
            log("FAILED:\n" + traceback.format_exc())
            yield "\n".join(lines), None, None, None, None
            return

        log("done")
        yield (
            "\n".join(lines),
            str(res["original_mp4"]) if res["original_mp4"] else None,
            str(res["edited_mp4"]) if res["edited_mp4"] else None,
            str(res["edited_pkl"]),
            res["info"],
        )

    with gr.Blocks(title="motion edit demo") as demo:
        gr.Markdown(
            "## tokenizer + infiller motion edit\n"
            "upload a WHAM `.pkl` (raw `wham_output.pkl` or a `wham_output_selected.pkl`) "
            "or an `.mp4` (WHAM runs first, slow). orange frames in the edited video are the "
            "masked frames the infiller rewrote."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_file = gr.File(
                    label="input (.pkl or .mp4)",
                    file_types=[".pkl", ".pth", ".mp4", ".mov", ".avi", ".mkv"],
                    type="filepath",
                )
                span = gr.Slider(0.05, 0.5, value=0.15, step=0.05,
                                 label="mask span fraction (around the kinematic peak)")
                splice = gr.Radio(
                    ["window", "mask"], value="window",
                    label="splice mode",
                    info="window = whole 90-frame window from the decoder; mask = only the masked frames",
                )
                window_only = gr.Checkbox(True, label="render only the 90-frame window (faster)")
                do_render = gr.Checkbox(True, label="render videos (uncheck for pkl only)")
                with gr.Row():
                    fps = gr.Number(30, label="fps", precision=0)
                    size = gr.Number(640, label="render size (px)", precision=0)
                run_btn = gr.Button("run pipeline", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    orig_vid = gr.Video(label="original", autoplay=True, loop=True)
                    edit_vid = gr.Video(label="edited", autoplay=True, loop=True)
                info = gr.JSON(label="run info")
                edited_pkl = gr.File(label="edited_motion_smpl.pkl")
                log_box = gr.Textbox(label="log", lines=14, max_lines=30)

        run_btn.click(
            run,
            inputs=[input_file, span, splice, window_only, fps, size, do_render],
            outputs=[log_box, orig_vid, edit_vid, edited_pkl, info],
        )

    return demo


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # must happen before torch import

    sys.path.insert(0, str(APP_DIR))
    import pipeline  # noqa: E402  (imports torch -> after CUDA_VISIBLE_DEVICES is set)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device: {pipeline.get_device()}")
    print(f"tokenizer ckpt: {pipeline.TOKENIZER_CKPT} (exists={pipeline.TOKENIZER_CKPT.exists()})")
    print(f"infiller ckpt:  {pipeline.INFILLER_CKPT} (exists={pipeline.INFILLER_CKPT.exists()})")
    print(f"smpl models:    {pipeline.SMPL_MODEL_DIR} (exists={pipeline.SMPL_MODEL_DIR.exists()})")

    demo = build_ui(pipeline)
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(RUNS_DIR)],
    )


if __name__ == "__main__":
    main()