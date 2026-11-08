"""Thin entry builders. Execution, checkpointing and aggregation stay in runtime."""
import ast
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

ENTRIES={
 'paper_suite_v1':('完整论文连续入口','run_colab','GPU'),
 'paper_fpr_validation_v1':('有限 FPR 连续入口','run_fpr_colab','GPU'),
 'native_baselines_public_compact12_v1':('公开 Baseline 十二视图连续入口','run_public_compact_colab','GPU'),
 'paper_parallel_worker_v1':('论文实验独立分片','run_parallel_colab','GPU'),
 'paper_finalize_v1':('唯一 CPU 汇总入口','run_finalize_colab',None),
}
PRESET_WORKERS={
 'paper_parallel_main0_v1':('主方法分片 0（含同 Wan 对照）','main',0),
 'paper_parallel_main1_v1':('主方法分片 1（含同 Wan 对照）','main',1),
 'paper_parallel_public0_v1':('公开 Baseline 分片 0','public',0),
 'paper_parallel_public1_v1':('公开 Baseline 分片 1','public',1),
 'paper_parallel_public2_v1':('公开 Baseline 分片 2','public',2),
 'paper_parallel_public3_v1':('公开 Baseline 分片 3','public',3),
}
ENTRIES.update({stem:(title,'run_parallel_colab','GPU') for stem,(title,role,shard) in PRESET_WORKERS.items()})


def build(stem,root=ROOT):
    title,function,accelerator=ENTRIES[stem];cells=[]
    def add(kind,text):
        cell=dict(cell_type=kind,id=stem.replace('_','-')+'-'+str(len(cells)),metadata={},source=text.splitlines(keepends=True))
        if kind=='code':ast.parse(text);cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('code',"from google.colab import drive\ndrive.mount('/content/drive')\n")
    descriptions={
      'paper_suite_v1':'单运行时顺序处理固定六片，主方法与同 Wan 对照、公开 Baseline 和最终 CPU 汇总都在 Colab。与任何子入口互斥。默认连续处理所有固定片；中断后再次 Run all 复用真实 checkpoint。',
      'paper_fpr_validation_v1':'包含主方法 GPU 工作：两片 C4 后执行原定随机查询和有限 FPR 汇总。与 main 分片和完整入口互斥；不跑同 Wan 对照。可与 public 分片同时跑计算，最终统计需等所有 worker 停止。',
      'native_baselines_public_compact12_v1':'一个 GPU 运行时顺序处理四片公开 Baseline；CAL32/TEST500、四配置、十二视图。不能与 public 分片或完整入口同时运行；可与两个 main 分片并行。最终用 CPU 汇总入口收集全表。',
      'paper_parallel_worker_v1':'最多六个独立 GPU runtime：main/0、main/1、public/0、public/1、public/2、public/3。每个组合只启动一次。main 每源先 C4 再同 Wan 对照；public 按完整 pack 分片，PixelSeal/FrameIndex 共享物理计算不拆开。',
      'paper_finalize_v1':'选择 CPU，无需 GPU。所有计算 worker 停止或安全暂停后，仅启动这一份：执行依赖已齐的原36次随机查询，并从保存记录生成全十二源/公开532源固定分母报告。缺失、失败和未决保留；依赖不齐时只给部分报告，不重抽查询。',
    }
    if stem in PRESET_WORKERS:
        _,role,shard=PRESET_WORKERS[stem]
        work='本片固定来源的 C4 和同 Wan 四方法对照' if role=='main' else '本片完整 pack 的公开 Baseline；PixelSeal/FrameIndex 共享计算保留在同片'
        descriptions[stem]=f"本入口已预填 `ROLE = '{role}'`、`SHARD = {shard}`，无需修改分工参数。选择独立 GPU runtime，挂载原 Google Drive 后 Run all，执行{work}。可与另外五个不同分片入口并行；相同分片只能运行一份，不与完整串行入口或重叠子入口同时运行。默认持续推进本片固定队列，不按时间或数量自动暂停。中断后在本入口再次 Run all，跳过已完成阶段，从当前未完成阶段的保存输入重新执行；前次中断记录和重算成本保留。所有计算 worker 停止或暂停后，单独运行[唯一 CPU 汇总](https://colab.research.google.com/github/RICHAAARC/SC-SSTW/blob/main/notebooks/paper_finalize_v1.ipynb)。"
    add('markdown','# '+title+'\n\n'+descriptions[stem]+'\n\n[运行顺序、分片、恢复和时长说明](https://github.com/RICHAAARC/SC-SSTW/blob/main/docs/paper_execution_v1.md)\n')
    params="SOURCE_REF = 'main'\nSOURCE_DIRECTORY = None  # 可选可读源码，不要求版本摘要一致\nPREVIOUS_RUNTIME_STOPPED = False  # 前一个占用本片的 runtime 已实际停止且换机时才填 True\n"
    if accelerator:params+="MAX_ITEMS = None  # 默认持续运行，不按源/pack数量自动暂停\nMAX_HOURS = None  # 默认持续运行，不设置人为时长暂停；原科学资源上限保留\n"
    if stem=='paper_parallel_worker_v1':params+="ROLE = 'main'  # main 或 public\nSHARD = 0  # main 为 0/1；public 为 0/1/2/3\n"
    elif stem in PRESET_WORKERS:params+=f"ROLE = {role!r}  # 本入口已预填，无需修改\nSHARD = {shard}  # 本入口已预填，无需修改\n"
    add('code',params)
    add('markdown','中断后保持 ROLE/SHARD、目录和方法不变，再次 Run all。先实际停止旧 worker；完整阶段直接复用，未完成阶段从保存输入重做。主方法与同 Wan 每源共享最多两次基础设施恢复，额外调用与耗时单列；原算法失败和旧终态失败不重试。公开 Baseline 沿用原恢复预算。\n\n这是阶段/任务级续跑，不承诺生成步、求解迭代或 VAE 内部任意位置无损恢复。最终汇总只有一个写入者。\n')
    call="source_directory=SOURCE_DIRECTORY, source_ref=SOURCE_REF, previous_runtime_stopped=PREVIOUS_RUNTIME_STOPPED"
    if accelerator:call+=", max_items=MAX_ITEMS, max_hours=MAX_HOURS"
    if function=='run_parallel_colab':call+=", role=ROLE, shard=SHARD"
    add('code',"from pathlib import Path\nimport urllib.request\nbootstrap_path = 'runtime/wan/paper_suite_colab_v1.py'\nif SOURCE_DIRECTORY:\n    bootstrap = (Path(SOURCE_DIRECTORY) / bootstrap_path).read_text()\nelse:\n    bootstrap = urllib.request.urlopen(f'https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/{SOURCE_REF}/{bootstrap_path}', timeout=120).read().decode('utf-8')\nentry = {'__name__': 'paper_suite_colab_entry'}\nexec(compile(bootstrap, bootstrap_path, 'exec'), entry)\nRUN = entry['"+function+"']("+call+")\n")
    add('markdown','正式结果与失败包均保留在 `MyDrive/Video-WM_Result`；可恢复大数组放独立 `MyDrive/Video-WM_Cache`，暂停时不删除。只有完整结束的自有临时缓存可以释放。分片仅改变调度，不改变 DCT2＋CDC32、旧 H/L、固定源/视图/分母、阈值或科学调用预算。\n\n入口返回、测量完成与科学结论分开；随机查询条件界不等于固定身份的总体 FPR 认证。公开自然视频与同 Wan 对照分表报告。\n')
    add('code',"print('入口状态:', RUN['status'])\nprint('本次记录:', RUN['invocation'])\nprint('各片返回:', [(row['stage'], row['result'].get('status')) for row in RUN['stages']])\nentry['show_results']()\n")
    metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'))
    if accelerator:metadata['accelerator']=accelerator
    path=Path(root)/'notebooks'/(stem+'.ipynb');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,metadata=metadata,cells=cells),ensure_ascii=False,indent=1)+'\n')
    return path


if __name__=='__main__':
    for stem in ENTRIES:print(build(stem))
