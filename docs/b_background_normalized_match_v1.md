# Card101 background-normalized matching

One frozen40-dimensional Q working model. No old L/F/V-min gate. Gamma1/writer/net32/default/A unchanged. See diagnostics/b-line-background-normalized-match-package-20261022/DESIGN.md for the pre-execution method checkpoint.

## Method

Every h has10observed rows×4ROIs, y_h[r,k]=Q_p[r,k,s+r-1]. Fit two key models only on assigned2cal OFF: host-equal global mean and full40covariance around finalglobal mean. The fixed covariance is(V+trace(V)/40 I)/2, not a scanned coefficient or a population whitening assertion. One unknown positive common-gain template ones40 and received-only whitened total norm define max(0,u'x)/(||u||||x||). Exact zero norm/nonfinite/singular inputs are technical; no pseudoinverse/fallback.

Background96Q all required; if any is missing no fit. Null threshold uses max of full137S for30null query percal host, then max(0,N1,N2). All60null families must be complete; no partial maximum. Positive cal scores are not an acceptance floor and not used for the threshold. TrainingOFF/null reuse can be optimistic; effective source units2, not3288 independent vectors. No CFAR, populationFPR, position coverage or role-randomization rank guarantee.

DetectionM>tau_N; evidence setC={S>=tau_N}. Rejection preservesC but reads nothing. Singleton unverified, multiC location unresolved; true-state exclusion and wrongsingleton remain possible. Earliest commoncell fixed before anybit access; ownphase/localrow of eachh, equality only, no rescoring by payload, no changedcell, candidate deletion or max fallback. Save all geometry, partial/mixed/guard/missing states. Nominal4frames is not causal VAE support proof.

## Entries

- User CLI: python -m experiments.wan_state_clock.b_background_normalized_match_v1_run
- User notebook: notebooks/b_background_normalized_match_v1.ipynb
- Builder: scripts/build_b_background_normalized_match_v1_notebook.py
- Ordinary source archive: notebooks/b_background_normalized_match_v1_user_bundle.zip
- Seen CPU development CLI: python -m experiments.wan_state_clock.b_background_normalized_cpu_v1_run (fixed local Card100Q; never fresh confirmation)

Fixed userDrive output /content/drive/MyDrive/Video-WM/B-Background-Normalized-Match-V1/fixed-four-unseen-v1. Fixed newsource IDs b101-s01..04,seeds2026102205..08,fullprompts in config. No source has been generated or role randomly assigned during preparation. Physical8fresh50chains/DiT800/native400/grad100,PM4/704constraints,VAE4/D12/E288/receiverFFT144,Q576,MP4encode/probe/decode12each. CalibrationQ-only; freeze background and nulltau before anyconfirmation sourceprep. Same physicalbudget asCard99; no extra model/E/FFT for matching. Full budgets in configs.

B0 uses newmatched statistic, unique-max1candidate/query, same tau, originalstrictzero/Counter ties. Ungated/gated share1reduction and both save before truth. OldCard100 B0 inCPU report is explicitly historical, with its oldscore/tau, not a newmatched B0 recomputation.

## Persistence and evaluation

Attempt/roles precede source observations. CalrawQ beforefit,calnumerical families beforetau; allfreeze beforeconfirm. All72new decisions and72B0pairs save successfully before post reads semantic bits; a failed save leavespost NOT_ENTERED. Legitimate rejection/emptycommon/bitdisagreement are method results, not technical failure. Actualmodel/Q/numerical/missing selectedraw failures remain technical and prevent top-levelCOMPLETE. Zero norm retains137-member unresolved family, never drops the zero candidate.

Existingattempt/roles/background/numeric/receipts means readonly, even without source/dependencies innotebook; no automatic physicalretry or outputdirectory switch. Notebook firstcodecell onlyDrive mount, then availableGitHubdev source, no SOURCE_TEXT/B64/hash/manifest/Git-clean/expectedversion check, no GPUwhitelist/resourcecap. Rebuild actualnotebook andZIP withbuilder whenever runner/entry wiring changes.

Confirmation72query/60null/12positive,9864H/2304message slots. Pos384=160physicalpresence+224missing; trueE41nominal144,present-unobserved16;120cell-view/960rawbit,84unique cell/672bit; net32/source. Keep allfailures and blanks; separateC coverage,commoncanonical support,actuallookup,emittedsourcebit,pointlocation,message coverage and old/newcorrectretain/loss/add. Report OFF/wrongkey/missinggroup/unsupported/wronggroup/crossgroup; do not filter by trueinC or count alternative windows as multiple true paths. PhysicalP0-PM rawbitBER is separate fromlogicalmatching benefit; no qualityPASS threshold.

## Scope

Card100 fourhosts allseen: CPUdiagnostic only, old96correct/48loss/6null/4empty andquality costs unchanged. A new fourtuple usertask is required to claim unseen execution, with roles persisted once uniformly before observations, but role randomness does not supply a rank guarantee after refitting background. Agent runs no physicalwork. Source/engineering and package readiness are not scientificPASS. Principal publishes only adoptedrelated files toexistingdev; no mainmerge.

## 已完成的固定CPU开发信号（全部旧来源已见）

一次867槽全部返回、19728统计量/72决定，无第二候选或物理调用。tau_N=.40451824225373706。正例12/12检出且真态入C，136/384位正确（Card100为96），相对Card100保留88/损失8/新增48；对旧gatedB0保留136/损失8/新增0。但null检出16/60（原6），null输出65，其中OFF49、缺失组25、canonical不支持65（类别交叉），错钥输出0。当前模型恢复部分正例覆盖，却明显劣化背景拒识；不认证位置、不替换默认、不声称科学改善。没有调参数/系数/阈值或换第二候选补救。新物理运行包是同一固定候选的用户复核任务，不自动执行，也不把未运行的新四来源写成确认成功。

正例共同支撑110/120cell，实际只选17/120，136/960原位bit读出、824未读。物理P0-PM仍仅复用Card100的960bit 0错→0错，没有新物理BER收益。所有负结果和历史质量代价独立保留。

开发null65的构成明确为OFF/key00 48、OFF/key01 1、P0/key00 16。OFF/key01也无嵌入目标，计OFF输出；“错钥归属0”仅指P0/PM key01。缺失组25、wrong-group33与以上类别交叉，不相加。
