"""Formal Wan terminal state-clock experiment; persists every fixed-denominator row."""
from __future__ import annotations
import argparse
import gc
import json
import subprocess
from pathlib import Path
from runtime.wan.generation import load_frozen_vae, generate_terminal_latent
from runtime.wan.vae import decode_normalized_latent, reencode_rgb24_readback
from runtime.wan.quality import rgb_quality_metrics
from runtime.wan.io import dump, read_mp4, encode_rgb
from main.tube_state import projection_margin as carrier
from main.tube_state import state_clock

ARMS = ("OFF", "MESSAGE_0", "MESSAGE_1")
def run(config: dict, output: Path) -> dict:
    import numpy as np
    import torch
    if config.get('protocol') != 'state_clock_v1':
        raise ValueError('formal runner requires protocol state_clock_v1')
    edit_frame = int(config.get('edit_source_frame',138))
    fresh = config.get('terminal_mode') == 'generate_new'
    conditions = ('RESAVED', f'DELETE{edit_frame}', f'REPEAT{edit_frame}')
    videos = tuple(a + '_NORMAL' for a in ARMS) + tuple(f'MESSAGE_{m}_{c}' for m in (0,1) for c in conditions)
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "RUNNING", "diagnostic_denominator": {"arms": 3, "received_videos": len(videos), "receiver_encodes": 4*len(videos)},
              "formal_science_denominator": 0, "fixed_calls": {"transformer": 2*config['generation']['steps'] if fresh else 0, "generation": int(fresh), "vae_decode": 3, "vae_encode": 4*len(videos)},
              "actual_calls": {"generation_attempted":0,"generation_completed":0,"transformer_attempted":0,"transformer_completed":0,"vae_decode_attempted": 0, "vae_decode_completed": 0, "vae_encode_attempted": 0, "vae_encode_completed": 0},
              "videos": {name: {"status": "NOT_RUN", "observations": {str(g): {"status": "NOT_RUN"} for g in range(4)}} for name in videos}, "failures": []}
    dump(output / "config.json", config)
    def save():
        dump(output / "result.json", result)
    def fail(stage, exc):
        result["failures"].append({"stage": stage, "error": repr(exc)})
        save()
    save()
    vae = None
    try:
        result["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if fresh:
            result['actual_calls']['generation_attempted']+=1
            save()
            def progress(counts):
                result['actual_calls'].update(counts)
                save()
                if counts['transformer_completed'] and counts['transformer_completed']%10==0 and counts['transformer_attempted']==counts['transformer_completed']:
                    print('generation Transformer forwards completed:',counts['transformer_completed'],flush=True)
            generated=generate_terminal_latent(config,progress=progress)
            terminal=generated.normalized_latent.detach().cpu().float()
            vae=generated.vae
            result['generation']=generated.metadata
            result['actual_calls']['generation_completed']+=1
            torch.save(terminal,output/'shared_terminal_normalized.pt')
            del generated
            gc.collect()
            torch.cuda.empty_cache()
            save()
        else:
            terminal = torch.load(config["source_shared_terminal"], map_location="cpu", weights_only=True).float()
        z = terminal.numpy()
        carrier.blocks(z)  # Checks actual data geometry, not an environment gate.
        book = state_clock.codebook(config['key_utf8'].encode())
        np.savez(output / "codebook.npz", **book)
        dump(output/'public_context.json', state_clock.CONTEXT)
        dump(output/'state_trajectories.json', {'context_digest':state_clock.CONTEXT_DIGEST,'states':book['states'].tolist(),'steps':book['steps'].tolist(),'message_drives':book['drives'].tolist()})
        result["protocol"] = {"namespace": carrier.NAMESPACE.decode(), "shape": list(z.shape), "support_count": 1760,
                              "tubelet_groups": 4, "patch": [4, 4], "margin": 1.0, "message_candidates": [0, 1],
                              "temporal_groups_written": [1, 44], "special_group_0_untouched": True, "tail_group_45_untouched": True,
                              "search_candidates_per_video": 204, "writer_support_score_weight": 0,
                              "normal_vs_resaved": "encoding-generation control", "delete_vs_resaved": "matched second lossy encoding"}
        result['protocol'].update(state_encoding='four circular phases -> two groups of 80 spatial projection supports',
                                  state_context=state_clock.CONTEXT, state_context_digest=state_clock.CONTEXT_DIGEST,
                                  global_candidates=204, local_clock_candidates=4284, observer_gain=.5, innovation_weight=.05, clock_event_cost=.002,
                                  aisb='not adopted: no shared affine projection channel established', flow_trajectory_modified=False)
        result['protocol']['edit_source_frame_zero_based']=edit_frame
        result['protocol']['event_window']=(edit_frame-1)//16
        save()
        if vae is None:
            vae = load_frozen_vae(config)
        off_rgb = None
        for arm in ARMS:
            try:
                if arm == "OFF":
                    marked = z.copy()
                else:
                    m = int(arm[-1])
                    marked, record = carrier.write(z, book, m)
                    dump(output / "write_evidence" / f"{arm}.json", record)
                dest = output / "write_evidence" / f"{arm}_terminal.pt"
                dest.parent.mkdir(exist_ok=True)
                torch.save(torch.from_numpy(marked), dest)
                result["actual_calls"]["vae_decode_attempted"] += 1
                save()
                latent = torch.from_numpy(marked).to(next(vae.parameters()).device)
                rgb = decode_normalized_latent(vae, latent).detach().cpu()
                result["actual_calls"]["vae_decode_completed"] += 1
                del latent, marked
                torch.save(rgb, output / "write_evidence" / f"{arm}_precodec_rgb.pt")
                if arm == "OFF":
                    off_rgb = rgb
                elif off_rgb is not None:
                    result.setdefault("precodec_quality_vs_off", {})[arm] = rgb_quality_metrics(off_rgb, rgb)
                name = arm + "_NORMAL"
                path = output / "received_videos" / f"{name}.mp4"
                encode_rgb(rgb, path, 8, 18)
                result["videos"][name].update(status="VIDEO_PERSISTED", path=str(path.relative_to(output)), expected_frames=181)
                del rgb
                save()
            except Exception as exc:
                result["videos"][arm + "_NORMAL"]["status"] = "FAILED_SOURCE"
                fail(arm, exc)
            gc.collect()
            torch.cuda.empty_cache()
        del off_rgb
        for m in (0, 1):
            source = output / "received_videos" / f"MESSAGE_{m}_NORMAL.mp4"
            try:
                rgb = read_mp4(source)
                if rgb.shape[0] != 181:
                    raise ValueError("normal MP4 must have 181 frames")
                for condition in conditions:
                    name = f"MESSAGE_{m}_{condition}"
                    try:
                        edited = rgb if condition == 'RESAVED' else (torch.cat((rgb[:edit_frame],rgb[edit_frame+1:]),dim=0) if condition.startswith('DELETE') else torch.cat((rgb[:edit_frame+1],rgb[edit_frame:edit_frame+1],rgb[edit_frame+1:]),dim=0))
                        path = output / "received_videos" / f"{name}.mp4"
                        encode_rgb(edited, path, 8, 18)
                        result["videos"][name].update(status="VIDEO_PERSISTED", path=str(path.relative_to(output)), expected_frames=int(edited.shape[0]))
                    except Exception as exc:
                        result["videos"][name]["status"] = "FAILED_EDIT"
                        fail(name, exc)
                del rgb, edited
            except Exception as exc:
                fail(f"MESSAGE_{m}_edit_source", exc)
            save()
        # All receiver observations use the same grid. Truth is never a read() input.
        for name in videos:
            item = result["videos"][name]
            obs = {}
            try:
                rgb = read_mp4(output / "received_videos" / f"{name}.mp4")
                n = int(rgb.shape[0])
                if n != item.get("expected_frames"):
                    raise ValueError("received length differs from persisted edit definition")
                item["received_frames"] = n
                for g in range(4):
                    dest = output / "receiver_observations" / name / f"g{g}"
                    dest.mkdir(parents=True, exist_ok=True)
                    groups, tail = divmod(n - g - 1, 4)
                    row = {"status": "RUNNING", "g": g, "frames_used": 1 + 4 * groups, "tail_discarded": tail}
                    item["observations"][str(g)] = row
                    try:
                        result["actual_calls"]["vae_encode_attempted"] += 1
                        save()
                        encoded = reencode_rgb24_readback(vae, rgb[g:g + 1 + 4 * groups]).detach().cpu().float()
                        result["actual_calls"]["vae_encode_completed"] += 1
                        torch.save(encoded, dest / "reencoded_normalized.pt")
                        if tuple(encoded.shape) != (1, 16, 1 + groups, 40, 64):
                            raise ValueError(f"unexpected receiver shape {encoded.shape}")
                        obs[g] = encoded.numpy()
                        row.update(status="COMPLETE", latent_shape=list(encoded.shape))
                        del encoded
                    except Exception as exc:
                        row.update(status="FAILED", error=repr(exc))
                        fail(f"{name}/g{g}", exc)
                    dump(dest / "observation.json", row)
                    save()
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"Wan projection: {name} g={g} {row['status']}", flush=True)
                del rgb
            except Exception as exc:
                fail(f"{name}/readback", exc)
            detection = state_clock.read(obs, book)
            dump(output / "detections" / f"{name}.json", detection)
            item["detection"] = {k: v for k, v in detection.items() if k not in ('candidates','classes')}
            item["status"] = "COMPLETE" if len(obs) == 4 else "PARTIAL_OR_FAILED"
            # Reporting only: label joins happen after blind search has finished.
            if name != "OFF_NORMAL":
                truth = int(name.split("_")[1])
                delta = 1 if name.endswith(f'DELETE{edit_frame}') else (-1 if name.endswith(f'REPEAT{edit_frame}') else 0)
                item['reporting_only'] = state_clock.report(detection, truth, delta,edit_frame)
            save()
            del obs
        result["status"] = "EXECUTED_REQUIRES_METHOD_REVIEW" if not result["failures"] else "EXECUTED_WITH_RETAINED_FAILURES"
    except Exception as exc:
        result["status"] = "FAILED"
        fail("setup_or_runner", exc)
    finally:
        del vae
        gc.collect()
        torch.cuda.empty_cache()
        save()
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(json.loads(args.config.read_text()), args.output)
    print(json.dumps({"status": result["status"], "actual_calls": result["actual_calls"], "failures": result["failures"]}, indent=2))
    if result["failures"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
