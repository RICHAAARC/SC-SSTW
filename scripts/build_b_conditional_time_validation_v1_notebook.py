"""Build the user-run c01 P0 materialization plus fixed 43-call validation notebook."""
from pathlib import Path
import ast
import json
import nbformat

ROOT=Path(__file__).resolve().parents[1]
STEM='b_conditional_time_validation_v1'
SETTINGS='''SOURCE = '/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1'
OUTPUT = '/content/drive/MyDrive/Video-WM/B-Conditional-Time-Validation-V1/fixed-c01-v1'
SOURCE_DIRECTORY = None  # Optional editable source; ordinary remote source is the default.
SOURCE_REF = 'dev/local-joint-state-payload-v1'
MODEL_ID = None  # Uses the existing Wan model location from the ordinary configuration.
'''
RUN='''from pathlib import Path
import importlib.util, json, sys, tempfile, urllib.request
if (Path(OUTPUT)/'attempt.json').exists():
    result = {'status': 'READ_ONLY_EXISTING_PACKAGE', 'physical_calls_this_invocation': 0, 'saved': {}, 'read_errors': {}}
    for name in ('attempt', 'state', 'counts', 'result'):
        try: result['saved'][name] = json.loads((Path(OUTPUT)/(name+'.json')).read_text())
        except (OSError, ValueError) as error: result['read_errors'][name] = str(error)
elif Path(OUTPUT).exists() and (not Path(OUTPUT).is_dir() or any(Path(OUTPUT).iterdir())):
    result = {'status': 'OUTPUT_CONFLICT', 'physical_calls_this_invocation': 0}
else:
    try:
        with tempfile.TemporaryDirectory(prefix='conditional-validation-bootstrap-') as temporary:
            root = Path(SOURCE_DIRECTORY) if SOURCE_DIRECTORY is not None else Path(temporary)
            if SOURCE_DIRECTORY is None:
                files = ('scripts/b_conditional_time_validation_v1_entry.py', 'scripts/b_conditional_time_output_v1_entry.py',
                         'scripts/b_framewise_qim_v1_entry.py', 'experiments/wan_state_clock/b_conditional_time_validation_v1_records.py',
                         'experiments/wan_state_clock/b_conditional_time_output_v1_records.py', 'experiments/wan_state_clock/b_minimal_l_v1_records.py',
                         'experiments/wan_state_clock/b_minimal_l_materialization_v1_records.py', 'experiments/wan_state_clock/b_framewise_qim_v1_records.py',
                         'experiments/wan_state_clock/b_framewise_qim_saved_candidate_v1_records.py', 'experiments/wan_state_clock/configs/b_conditional_time_validation_v1.json')
                for relative in files:
                    path = root/relative; path.parent.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve('https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/'+SOURCE_REF+'/'+relative, path)
            spec = importlib.util.spec_from_file_location('conditional_validation_entry', root/'scripts/b_conditional_time_validation_v1_entry.py')
            entry = importlib.util.module_from_spec(spec)
            previous = sys.dont_write_bytecode; sys.dont_write_bytecode = True
            try: spec.loader.exec_module(entry)
            finally: sys.dont_write_bytecode = previous
            result = entry.launch(SOURCE, OUTPUT, SOURCE_DIRECTORY, SOURCE_REF, MODEL_ID)
    except Exception as error:
        result = {'status': 'PREPARATION_FAILED', 'error': str(error), 'physical_calls_this_invocation': 0}
print(result)
'''


def build(root=ROOT):
    cells=[nbformat.v4.new_code_cell("from google.colab import drive\ndrive.mount('/content/drive')\n"),
           nbformat.v4.new_markdown_cell('''# c01 新时间包物化与条件时间输出验证
Run all 使用既有 b2-m3-c01 身份P0，顺序执行原 Minimal-L A/B 时间包物化，再运行已审查V3条件接口的固定43项媒体验证。默认源按原c01、c02顺序选择c01：两者用于历史身份校准，但在已核对的当前融合L/条件接口设计记录中未使用；不把它称为全研究未见内容。不会重新DiT、训练或改名e02。源码默认普通下载并按实际导入错误准备依赖；无需填写SOURCE_DIRECTORY，没有摘要/manifest/Git-clean/精确版本门。

整包先记录唯一attempt，再开始A/B/接收。重新Run all一律只读，包括上次只完成A、失败、中断或尚未开始B/接收；只允许同一次顺序调用推进尚未消费的后续阶段。旧输出独立保留。A需要完整有限缓存；B保留直接约束与native数值失败。native有限完整、QIM媒体文件存在时可进入固定43项读取，真实可读性由其预算内probe/decode检查；接收结果不把物化失败改写为成功。没有自动重试或备用源。

物化独立预算：model_load2、D1811、BASE head46、QIM head46、solver1、targets1、K4201/KT4200、BASE/Aa投影各1、四层时间读8（912240位槽）、codec6。沿原4000迭代/每20检查、7200秒、16GiB求解与1e-3目标/1e-7预算容差；原最优性UNPROVEN保持。缓存需按实际head通道准备足够磁盘（现有路径至少32GiB）；GPU型号不设门。

验证固定一个正例五视图：FULL181、CROP[38,139)101、删44为180、重复44为182、删43/44/45为178；共822节点/817边/517860位槽/4110lane。43项=源probe/decode2+码表1+每view encode/probe/decode/time_read/hard/C/L/提交各1；其中codec17，整链codec23。接收RSS8GiB、码表300秒、每view600秒、累计3300秒，codec每次300秒另记。原双副本融合L、DCT2＋CDC32、旧H/旧边保留；全盲输出终结保存后才truth。

结果在OUTPUT的materialization/A、materialization/B和receiver，顶层result/state/counts分别汇总，不混合物化失败和接收判据。零错零未知且同一非退化新增正确节点参与新增正确边才可能获得本批新增部分有限支持；全拒/无增益不是PASS，顺序不认证绝对定位，单源不证明总体可靠性、Ø或身份闭环。工程准备尚未执行真实物化/媒体/模型/Colab；由用户运行。'''),
           nbformat.v4.new_code_cell(SETTINGS),nbformat.v4.new_code_cell(RUN)]
    for i,cell in enumerate(cells):
        cell['id']=f'conditional-validation-{i:02}'
        if cell.cell_type=='code':ast.parse(cell.source);cell.execution_count=None;cell.outputs=[]
    notebook=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3'),accelerator='GPU'))
    nbformat.validate(notebook)
    path=Path(root)/'notebooks'/(STEM+'.ipynb');path.parent.mkdir(parents=True,exist_ok=True);nbformat.write(notebook,path)
    return dict(notebook=str(path),scientific_cells_executed=False)


if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False))
