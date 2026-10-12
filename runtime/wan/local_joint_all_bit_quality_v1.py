"""All-frame float quality and shared actual-peak neighborhoods; no quality gate."""
from __future__ import annotations

import json
from pathlib import Path

from main.tube_state import local_joint_all_bit_v1 as method
from runtime.wan.local_joint_readout_m0_v1 import quality as original_quality
from runtime.wan.local_joint_terminal_bridge_v1 import write_json

DISPLAY_LIMIT = .25
CROP_RADIUS = 16


def peak_pixel(difference):
    import numpy as np
    flat = int(difference.abs().argmax())
    y,x,c = (int(v) for v in np.unravel_index(flat,tuple(difference.shape)))
    return dict(y=y,x=x,channel=c,max_abs=float(difference[y,x,c].abs()))


def quality(rgb,base,protocol):
    report = original_quality(rgb,base,protocol,include_temporal_framewise=True)
    previous = None
    for t,(frame,reference) in enumerate(zip(rgb,base)):
        difference = frame.double()-reference.double()
        peak = peak_pixel(difference)
        report["framewise"][t].update(frame=t,max_abs=peak["max_abs"],peak_pixel=peak,
            display_clipped_components=int((difference.abs()>DISPLAY_LIMIT).sum()),
            display_clipped_pixels=int((difference.abs()>DISPLAY_LIMIT).any(dim=-1).sum()))
        if previous is not None:
            report["temporal_framewise"][t-1]["peak_pixel"] = peak_pixel(difference-previous)
        previous = difference
    frame_peak = max(report["framewise"],key=lambda row:row["max_abs"])
    report["peak_pixel"] = dict(frame=frame_peak["frame"],**frame_peak["peak_pixel"])
    report["temporal_residual_max_abs"] = max(x["max_abs"] for x in report["temporal_framewise"])
    report["display"] = dict(rgb_range=[0,1],residual_range=[-DISPLAY_LIMIT,DISPLAY_LIMIT],
        numeric_metrics_unclipped=True, clipped_components=sum(x["display_clipped_components"] for x in report["framewise"]),
        clipped_pixels=sum(x["display_clipped_pixels"] for x in report["framewise"]))
    events=[]
    for kind,rows,field in (("frame_rmse",report["framewise"],"rmse"),
                            ("frame_max_abs",report["framewise"],"max_abs"),
                            ("temporal_rmse",report["temporal_framewise"],"rmse"),
                            ("temporal_max_abs",report["temporal_framewise"],"max_abs")):
        # Python max retains the first row at a tie. Zero residual adds no arbitrary peaks.
        item=max(rows,key=lambda row:row[field])
        if item[field] == 0:
            continue
        start=item.get("from_frame",item.get("frame"))
        end=item.get("to_frame",item.get("frame"))
        events.append(dict(kind=kind,value=item[field],from_frame=start,to_frame=end,
            neighborhood=list(range(max(0,start-1),min(len(rgb)-1,end+1)+1)),
            peak_pixel=item["peak_pixel"],pixel_quantity="temporal residual change" if start!=end else "residual"))
    report["peak_events"]=events
    report["peak_tie_rule"]="earliest index for visualization only"
    return report


def selection(reports,frame_count, *, views=method.VIEWS):
    frames={t for t in method.FRAMES if 0<=t<frame_count}
    crops=[]
    for name in views[1:]:
        for event in reports.get(name,{}).get("peak_events",[]):
            frames.update(event["neighborhood"])
            # Temporal events crop both endpoints, at the peak change pixel.
            for frame in sorted({event["from_frame"],event["to_frame"]}):
                crops.append(dict(origin_view=name,kind=event["kind"],frame=frame,**event["peak_pixel"]))
    return sorted(frames),crops


def render_pair(path,name,index,frame,reference, *, box=None):
    from PIL import Image,ImageDraw
    if box is not None:
        y0,y1,x0,x1=box
        frame,reference=frame[y0:y1,x0:x1],reference[y0:y1,x0:x1]
    residual=frame.double()-reference.double()
    clipped=int((residual.abs()>DISPLAY_LIMIT).any(dim=-1).sum())
    rgb=(frame.clamp(0,1)*255).round().byte().numpy()
    delta=((residual/(2*DISPLAY_LIMIT)+.5).clamp(0,1)*255).round().byte().numpy()
    h,w=frame.shape[:2]
    canvas=Image.new("RGB",(max(2*w,440),h+48),"white")
    canvas.paste(Image.fromarray(rgb),(0,48));canvas.paste(Image.fromarray(delta),(w,48))
    draw=ImageDraw.Draw(canvas)
    draw.text((0,0),f"{name} frame {index}; RGB [0,1]",fill="black")
    draw.text((0,15),f"Residual [-0.25,0.25]; clipped pixels={clipped}",fill="black")
    if box is not None: draw.text((0,30),f"crop y0,y1,x0,x1={box}",fill="black")
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    canvas.save(temporary,format="PNG")
    temporary.replace(path)
    return dict(path=path.relative_to(path.parent.parent).as_posix(),display_clipped_pixels=clipped,box=box)


def export_visuals(output,base, *, views=method.VIEWS):
    """Load one saved view at a time, retaining only BASE and that view on CPU."""
    import torch
    output=Path(output)
    reports={name:json.loads((output/name/"quality.json").read_text())
             for name in views if (output/name/"quality.json").exists()}
    frames,crops=selection(reports,len(base),views=views)
    index=dict(frame_indices=frames,crop_requests=crops,views={},failures=[],
        rgb_range=[0,1],residual_range=[-.25,.25],numeric_metrics_unclipped=True,
        coordinates="zero-based; crop upper bounds exclusive",
        path_base="each view directory: <run>/<view>/<path>")
    for name in views:
        path=output/name/"float_rgb.pt"
        if not path.exists():
            index["views"][name]=dict(status="MISSING",reason="float RGB missing")
            continue
        try:
            saved_index=output/name/"frames.json"
            if saved_index.exists():
                try:
                    previous=json.loads(saved_index.read_text())
                    from PIL import Image
                    if previous["frame_indices"]==frames and previous.get("crop_requests")==crops:
                        for item in previous["frames"]+previous["crops"]:
                            with Image.open(output/name/item["path"]) as image:
                                image.verify()
                        index["views"][name]=previous
                        continue
                except (OSError,ValueError,KeyError,TypeError):
                    pass
            rgb=base if name=="BASE" else torch.load(path,map_location="cpu",weights_only=True)
            images=[]
            for t in frames:
                images.append(render_pair(output/name/"frames"/f"frame_{t:03d}.png",name,t,rgb[t],base[t]))
            crop_images=[]
            for n,request in enumerate(crops):
                t,y,x=request["frame"],request["y"],request["x"]
                box=(max(0,y-CROP_RADIUS),min(rgb.shape[1],y+CROP_RADIUS+1),
                     max(0,x-CROP_RADIUS),min(rgb.shape[2],x+CROP_RADIUS+1))
                crop_images.append(render_pair(output/name/"crops"/f"crop_{n:02d}_{t:03d}.png",name,t,rgb[t],base[t],box=box))
            view=dict(status="SAVED",frame_indices=frames,frames=images,crops=crop_images,crop_requests=crops)
            write_json(output/name/"frames.json",view)
            index["views"][name]=view
            del rgb
        except Exception as exc:
            index["views"][name]=dict(status="FAILED",reason=repr(exc))
            index["failures"].append(dict(view=name,reason=repr(exc)))
    write_json(output/"visuals.json",index)
    return index
