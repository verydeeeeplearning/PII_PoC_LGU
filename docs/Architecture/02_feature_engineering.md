## 6. Stage S2: Feature Prep

### 6.1 기능

S1에서 생성된 정규화 데이터에 텍스트/경로/Tabular 피처를 추가한다. 이 단계에서 생성되는 피처가 RULE 라벨러와 ML 라벨러 모두의 공통 입력이 된다.

### 6.2 S2-1: raw_text + shape_text 생성

**raw_text:** 소문자화 + 공백 정리, 특수문자 유지, **고엔트로피 토큰만 placeholder 치환**

- 입력: `local_context_raw` (권장) + 필요시 `masked_hit` 일부
- 처리: lower + 공백 정리, 특수문자 유지(`@ . / = : - _ *` 등)
- `*` 연속은 `<MASK>`로 축약(권장) + 길이 정보는 별도 피처로 분리
- 고엔트로피 토큰만 placeholder로 치환:
  - `1234567890` → `<NUM10>` (숫자열 + 길이 정보 유지)
  - `1234567890123` → `<NUM13>`
  - `20240219` → `<DATE8>` (경로/컨텍스트에서 날짜 패턴)
  - `0x7ffde…` → `<HEX>`
  - `a94a8fe5…(32+자)` → `<HASH>`
  - `****` → `<MASK>`

**shape_text:** 숫자→`0`, 영문→`a`, 한글→`가`, `*` 유지, 구분자 유지

```python
def make_shape_text(text: str) -> str:
    """
    텍스트의 '형태(shape)'를 추출.
    내용이 마스킹으로 사라진 환경에서, 구조적 패턴이 핵심 신호가 된다.
    """
    result = []
    for ch in text:
        if ch.isdigit():
            result.append('0')
        elif ch.isascii() and ch.isalpha():
            result.append('a')
        elif '\uAC00' <= ch <= '\uD7A3':  # 한글
            result.append('가')
        else:
            result.append(ch)  # 특수문자, *, @, -, . 등 유지
    return ''.join(result)
```

**예시:**
| full_context | raw_text | shape_text |
|-------------|----------|------------|
| `xpiryDate=170603*****` | `xpirydate=170603*****` | `aaaaaaaaa=000000*****` |
| `****@bdp.lguplus.co.kr` | `****@bdp.lguplus.co.kr` | `****@aaa.aaaaaaa.aa.aa` |
| `45 bytes 141022******` | `45 bytes 141022******` | `00 aaaaa 000000******` |

**Rationale:**

- 마스킹 환경에서는 "내용(로컬 파트)"이 사라지고 "형식"이 핵심 신호가 된다. raw_text는 도메인/키워드(redhat, bytes, lguplus)를 잡고, shape_text는 버전/epoch/숫자열의 구조적 패턴을 잡는다.
- 두 가지를 같이 사용하면 짧은 컨텍스트(약 10자)에서 얻을 수 있는 정보량을 최대화할 수 있다. 이것은 "동일 원문에서 서로 다른 관점의 피처를 추출하는" 접근이며, Boosting 모델의 트리 분기에서 두 관점이 자연스럽게 상호작용한다.
- **표본 밖 패턴에 대한 방어:** raw_text는 알려진/신규 키워드를 흡수하고, shape_text는 키워드가 없어도 구조만으로 분류를 버틴다. 새 패턴이 들어와도 최소 하나의 뷰가 유효한 신호를 제공한다.
- **엔트로피 축소의 효과:** raw_text에서 고엔트로피 값(UUID, 해시, 긴 숫자열)만 placeholder로 치환하면, 단어 사전 폭발/과적합을 막으면서 구조/키워드 신호는 유지된다. placeholder에 길이 정보를 포함(`<NUM10>`, `<NUM13>`)하여 "숫자열 길이"라는 판별력은 보존한다.

### 6.3 S2-2: path_text 생성 (경로의 텍스트화)

파일 경로를 `/`, `.`, `_`, `-` 기준으로 쪼개 토큰화한 문자열을 생성한다. 이 텍스트는 TF-IDF 입력으로 활용 가능하다.

```python
def make_path_text(file_path: str) -> str:
    """
    파일 경로를 토큰화된 텍스트로 변환.
    예: /var/log/hadoop/... → var log hadoop ...
    """
    import re
    fp = file_path.lower() if file_path else ''
    # 구분자로 쪼개기
    tokens = re.split(r'[/\\_.\-]+', fp)
    # 빈 토큰 제거
    tokens = [t for t in tokens if t]
    return ' '.join(tokens)
```

**Rationale:**

- 경로 토큰은 word TF-IDF 또는 단순 키워드 플래그로 활용 가능하다.
- "var log hadoop"처럼 토큰화하면 시스템/인프라 경로 vs 업무 데이터 경로 패턴이 TF-IDF에서 자연스럽게 분리된다.
- (선택) `shape_path_text`도 가능하지만 PoC에서는 과할 수 있다.

파일 경로는 "텍스트 10자"만 보면 애매한 케이스에서 "생성 맥락"을 강하게 반영하는, 사실상 가장 강력한 판별 피처 중 하나다.

### 6.3a Phase 1 피처 보강 — label-only 모드 전용 (Wave 2 구현, Wave 3 개정)

Phase 1(label-only)은 `full_context_raw` 텍스트가 없다. `use_phase1_tfidf=True` 옵션으로 TF-IDF 피처를 추가하고, `build_meta_features()` + `extract_path_features()` 결과 컬럼을 feature matrix에 포함시킨다.

| 피처 그룹 | 소스 | 피처 수 | 비고 |
|----------|------|--------|------|
| Phase 1 TF-IDF (fname char + **fname shape** + path word) | file_name, file_path | ~500 | Wave 3: 500→200 축소, **Wave 4: shape 100 추가** |
| Dense — create_file_path_features | file_path | 8 | |
| Dense — build_meta_features (시간 피처 제외) | 메타 컬럼 | 8 | Wave 3: 시간 4개 제거 |
| Dense — extract_path_features (비중복 항목) | file_path | **10** | is_log_file, is_docker_overlay, has_license_path, is_temp_or_dev, is_system_device, is_package_path, has_cron_path, has_date_in_path, has_business_token, has_system_token |
| Dense — 운영 메타데이터 | rule_matched | 1 | ~~exception_requested~~: 제거 (Sumologic에 없음, 추론 불가) |
| **Dense — 서버 의미 토큰** | server_name | **3** | **Tier 2 B7: server_env(prd/dev/stg/sbx/test/unknown), server_is_prod, server_stack(app/mms/db/web/batch/etc)** |
| **Dense — RULE 세부 신호** | RuleLabeler | **3** | **Tier 2 B8: rule_confidence_lb + rule_id_enc + rule_primary_class_enc** |
| **Dense — file-level aggregation** | pk_file 집계 | **2** | **Tier 2 B9: file_event_count, file_pii_diversity (df에 추가, X_train 미주입)** |
| **Dense — 범주형 Label Encoding** | 메타 컬럼 | **8** | **Tier 2 B1: train+test 합본 fit. encoder가 FeatureBuilderSnapshot에 포함** |
| **합계 (중복 제거 후)** | | **~538 (실측)** | Tier 3 C1+C2: exception_requested 제거, F1=0.78 |

> **Wave 3 피처 개정 (2026-03-20):** 10M행 실데이터 학습 결과(F1-macro=0.6146) 진단 후, 과적합/누수 피처를 정리하고 강신호 피처를 추가. 상세: `outputs/diagnosis/model_performance_report.md`
>
> **제거된 피처 (5개):**
> - `created_hour`, `created_weekday`, `is_weekend`, `created_month` — 시간대/요일/월은 운영 환경 패턴이지 FP/TP 본질 아님. `created_hour`가 feature importance 1위(22%)였으나, temporal split에서 일반화 불가능한 과적합 신호. `created_month`는 temporal split 분할 기준과 직접 상관하여 split 자체를 학습하는 위험.
> - `server_freq` — train set의 서버 출현 빈도를 피처로 사용하면 test에 train 통계가 누수됨. 새로운 서버에 대해 0으로 fallback하여 불안정.
>
> **추가된 피처 (2개 → Wave 5에서 1개로 축소):**
> - ~~`exception_requested`~~ — Wave 3에서 추가, **Wave 5에서 제거** (Sumologic에 없어 추론 불가).
> - `rule_matched` — Step 4 RuleLabeler 결과(0/1). L3 룰 9개의 도메인 지식을 ML 피처로 전달.
>
> **TF-IDF 축소:** Phase 1 fname/path 각 `max_features` 500→200. 기존 1000개 TF-IDF 중 97%가 importance=0이었으며, 유효 피처만 남겨 노이즈 감소.

> **Wave 4 Tier 2 피처 개정 (2026-03-20):** Tier 0+1 실험 결과(F1=0.6114) + 독립 에이전트 코드 검증(`08_performance_improvement_playbook.md`) 반영. 상세: `model_performance_report.md v5`
>
> **B2 중복 샘플 가중치:** `(file_path, file_name)` 그룹별 `sample_weight = 1/sqrt(group_size)`, mean=1 정규화. 10M행에서 동일 파일 반복 행의 모델 편향 방지. `train_lightgbm(sample_weight=)` 파라미터 추가, early stopping 내부 split 시에도 가중치 동시 분할.
>
> **B7 서버 의미 토큰 (server_freq 대체):** `build_meta_features()`에서 `server_name` → `server_env`(prd/dev/stg/sbx/test/unknown), `server_is_prod`(0/1), `server_stack`(app/mms/db/web/batch/etc) 추출.
>
> **B1 범주형 Label Encoding:** `service`, `ops_dept`, `organization`, `retention_period`, `server_env`, `server_stack`, `rule_id`, `rule_primary_class` → `_enc` 접미사 정수 변환. train+test 합본 fit. encoder가 `build_features()` 반환 dict에 포함되어 `FeatureBuilderSnapshot`으로 전달 → 추론 시 동일 encoding 재현.
>
> **B8 RULE 세부 신호:** `rule_matched` binary 1bit에서 확장. `rule_id_enc` (12개 룰 구분), `rule_primary_class_enc` (FP 클래스 방향), `rule_confidence_lb` (Bayesian 하한 수치). Step 4에서 기존 버려지던 `rule_confidence_lb` 컬럼 복원.
>
> **B9 file-level aggregation:** `compute_file_aggregates_label(df_train)` → `file_event_count` (pk_file당 행 수), `file_pii_diversity` (SSN/Phone/Email 검출 유형 수 0~3). **train fold에서만 계산**, test는 left join + median fallback (누수 방지).
>
> **Tier 2 실측 결과:** 10M행 서버 학습, F1-macro=**0.8063** (Baseline 0.6146 대비 +0.19). Feature importance top 3: service_enc(2458), ops_dept_enc(2431), retention_period_enc(2050) — B1 범주형 피처가 주도.

> **Wave 5 Tier 3 (2026-03-21):** Sumologic 추론 가능성 검증 후 피처 정리 + 아키텍처 개선.
>
> **exception_requested 제거:** Sumologic 데이터에 해당 컬럼이 없어 추론 시 사용 불가. Wave 3에서 추가 → Wave 5에서 제거.
>
> **C1 Easy FP Suppressor:** ML 학습/평가 전 고확신 FP를 규칙 기반으로 선제 분리. 4개 조건 (is_system_device=1, is_package_path+is_mass_detection, is_docker_overlay=1, has_license_path=1). Train에서 purity≥95% 확인 시에만 활성화. Suppressed 행은 ML이 보지 않고 FP로 판정. 최종 평가에서 suppressed + ML residual을 합산하여 F1 출력.
>
> **C2 Slice-aware threshold:** server_env별 Coverage-Precision Curve tau를 개별 계산. `threshold_policy.json`에 `slice_thresholds` 딕셔너리로 저장. 운영 시 서버 환경별 차등 tau 적용 가능.
>
> **run_report.py 체크포인트 방식으로 전환:** 피처 재생성/split 재실행 제거. `step5_features_*.pkl` + `step6_model_*.pkl`에서 X_test/y_test_enc/model/le를 직접 로드하여 training과 100% 동일한 평가. `threshold_policy.json` 우선 로드.

#### 학습/추론 동형성 개선

> **공통 피처 준비 함수 (`feature_preparer.py`):** `run_training.py` Steps 2-4의 공통 로직 (build_meta_features + extract_path_features + RuleLabeler)을 `prepare_phase1_features(df)` 단일 함수로 추출. training과 inference(FeatureBuilderSnapshot.transform 내부)가 동일 함수를 호출하여 피처 동형성을 보장.
>
> **FeatureBuilderSnapshot:** `categorical_encoders` (LabelEncoder) 딕셔너리 저장. `transform()` 내부에서 `prepare_phase1_features()` 자동 호출 + 저장된 encoder로 categorical encoding 수행. 추론 시 Importance Top 5 피처(service_enc 등)가 NaN이 되던 문제 해결.
>
> **`run_inference.py`:** `MLFeatureBuilder` → `FeatureBuilderSnapshot` 전환. transform() 내부에서 전처리 자동 수행.
>
> **B3 Shape TF-IDF:** file_name에 `_to_shape_text()` 변환 후 char_wb n-gram TF-IDF 100 features 추가. 숫자 n-gram 과적합 감소: `02506` → `DDDDD` 패턴 학습으로 OOV 파일명 내성 증가.
>
> **제거된 항목:** CalibratedClassifierCV(cv=3) — F1 무관, 학습 시간만 증가. Bootstrap CI(n=500) — 4M행에서 ±0.0009 수준으로 실용성 없음.
>
> **B6 LightGBM 정규화 강화:** `min_child_samples: 20→200` (리프당 최소 샘플 수 증가), `reg_alpha: 0→0.5` (L1 정규화, 노이즈 피처 억제), `max_depth: -1→10` (트리 깊이 제한).
>
> **C5 threshold_policy.json:** Step 6c Coverage-Precision Curve 결과를 `models/final/threshold_policy.json`에 자동 저장. 추천 tau, curve summary, split 전략, F1 등 포함. decision_combiner의 하드코딩 임계값을 운영 정책으로 연결하는 기반.

> **Split 전략 확장 (2026-03-20):** `build_features()`에 `split_strategy` 파라미터 추가. `"group"` (기본 GroupShuffleSplit), `"temporal"` (label_work_month 시간 분할), `"server"` (server_name 서버 분할). `--test-months N`으로 temporal holdout 월 수 지정. 일반화 성능 검증에 temporal split 사용 권장.

> **Temporal Split 개정 (2026-03-20):** `work_month_time_split()`이 pk_file별 최대 월 기준 파일 단위 분할에서, **행 단위 엄격 월 분할**로 변경됨. 3~9월→train, 10~12월→test로 완전 분리. 동일 pk_file이 train/test 양쪽에 나타날 수 있으나 시간적 분리가 깨지지 않음.

> **Wave 2 버그 수정 (2026-03-19):** `build_meta_features` + `extract_path_features` 결과가 df에는 있었으나 `build_features()` 내부 `else` 분기에서 feature matrix에 포함되지 않았음. `_PRECOMPUTED_DENSE_COLS` 목록으로 명시적 포함 + 중복 컬럼 제거 추가.

| 피처 | 방법 | 차원 | 비고 |
|------|------|------|------|
| `phase1_fname` char TF-IDF | `TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), max_features=200)` on `file_name` | ~200 | Wave 3: 500→200 축소 |
| **`phase1_fname_shape`** char TF-IDF | `TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), max_features=100)` on `_to_shape_text(file_name)` | ~100 | **Wave 4 B3: 숫자→D 변환으로 과적합 감소** |
| `phase1_path` word TF-IDF | `TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=200)` on `_to_path_text(file_path)` | ~200 | Wave 3: 500→200 축소 |
| ~~`exception_requested`~~ | ~~예외 신청 여부 (Y→1, N→0)~~ | ~~1~~ | ~~Wave 3 추가~~ → **Wave 5 제거 (Sumologic에 없음)** |
| `rule_matched` | Step 4 RuleLabeler 매칭 여부 (0/1) | 1 | Wave 3 추가 |
| **`server_is_prod`** | server_name에서 prd 토큰 존재 여부 | 1 | **Wave 4 B7** |
| **`rule_confidence_lb`** | Bayesian precision 하한 | 1 | **Wave 4 B8** |
| **`file_event_count`** | pk_file당 행 수 (train fold only) | 1 | **Wave 4 B9** |
| **`file_pii_diversity`** | PII 유형 수 0~3 (train fold only) | 1 | **Wave 4 B9** |
| **범주형 `_enc` 8개** | Label Encoding (service, ops_dept, org, retention, server_env/stack, rule_id/class) | 8 | **Wave 4 B1** |
| ~~`server_freq`~~ | ~~train fold 서버 빈도~~ | ~~1~~ | **Wave 3 삭제.** → Wave 4 B7로 대체 |

```python
# Phase 1 TF-IDF 구성 (build_features 내부) — Wave 3 개정
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. file_name char n-gram (파일명 패턴) — max_features 500→200
tfidf_fname = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(2, 5), max_features=200,
    sublinear_tf=True, min_df=2, max_df=0.98,
)
X_fname_train = tfidf_fname.fit_transform(df_train['file_name'].fillna(''))
X_fname_test  = tfidf_fname.transform(df_test['file_name'].fillna(''))

# 2. file_path word TF-IDF (_to_path_text 전처리 후) — max_features 500→200
tfidf_path = TfidfVectorizer(
    analyzer='word', ngram_range=(1, 2), max_features=200,
    sublinear_tf=True, min_df=2, max_df=0.95,
)
X_path_train = tfidf_path.fit_transform(df_train['file_path'].fillna('').apply(_to_path_text))
X_path_test  = tfidf_path.transform(df_test['file_path'].fillna('').apply(_to_path_text))

# 3. server_freq — Wave 3에서 삭제됨 (train 통계 누수 위험)
# 4. rule_matched (0/1) — Wave 3 추가
# exception_requested — Wave 5에서 제거 (Sumologic에 없음)
#    _PRECOMPUTED_DENSE_COLS에 포함, run_training.py Step 4 이후 int 변환
```

> **구현 위치:** `src/features/pipeline.py` — `use_phase1_tfidf=True` 블록
> **저장:** fitted 벡터라이저는 `FeatureBuilderSnapshot`에 포함되어 `models/final/feature_builder.joblib`로 저장

```python
def extract_path_features(file_path: str) -> dict:
    """
    파일 경로에서 구조적 피처를 추출.
    경로는 텍스트 컨텍스트가 애매할 때 '파일 유형 맥락'을 제공한다.
    """
    fp = file_path.lower() if file_path else ''
    features = {}
    
    # 경로 구조
    features['path_depth'] = fp.count('/')
    features['extension'] = fp.rsplit('.', 1)[-1] if '.' in fp else 'unknown'
    
    # 파일 유형 플래그
    features['is_log_file'] = int(
        fp.endswith('.log') or '.log.' in fp or '/log/' in fp or '/logs/' in fp
    )
    features['is_docker_overlay'] = int(
        'overlay' in fp or 'docker' in fp or '/var/lib/docker/' in fp
    )
    features['has_license_path'] = int(
        any(t in fp for t in ['/usr/share/doc/', '/license', '/copyright', 'changelog'])
    )
    features['is_temp_or_dev'] = int(
        any(t in fp for t in ['/tmp/', '/test/', '/dev/', '/sample/', '/mock/', '/sandbox/'])
    )
    features['has_date_in_path'] = int(bool(re.search(r'/\d{8}/', fp)))
    
    # 업무 관련 토큰 (TP 방향 신호)
    features['has_business_token'] = int(
        any(t in fp for t in ['customer', 'crm', 'hr', 'billing', 'export', 'report', 'contact'])
    )
    # 시스템/인프라 토큰 (FP 방향 신호)
    features['has_system_token'] = int(
        any(t in fp for t in ['var/log', 'proc', 'kernel', 'auth', 'krb', 'kerberos', 'hadoop'])
    )
    
    return features
```

**Rationale:**

- 이 문제에서 텍스트 컨텍스트가 약 10자로 제한되어 있으므로, 파일 경로가 "생성 맥락"을 대리하는 역할을 한다. 동일한 `****@gmail.com`이라도 Hadoop 로그 파일에서 발견됐는지, CRM 엑셀에서 발견됐는지에 따라 FP/TP 확률이 극적으로 달라진다.
- 경로 기반 근거는 사람이 이해하기 쉬워서 설명가능성에도 매우 유리하다. evidence에 "파일이 /var/log/hadoop/ 경로에 있음"이라고 적으면 운영팀이 즉시 납득할 수 있다.
- Docker overlay 환경에서의 대규모 오탐 집중(문제 정의서 핵심 발견사항)을 직접적으로 포착하는 피처다.

### 6.4 S2-4: Tabular 피처 정규화

```python
def extract_tabular_features(row: dict) -> dict:
    features = {}
    
    # 검출 건수 (극단값이 매우 크므로 log 변환 필수)
    count = row.get('inspect_count', 0)
    features['inspect_count_raw'] = count
    features['inspect_count_log1p'] = np.log1p(count)
    features['is_mass_detection'] = int(count > 10000)
    features['is_extreme_detection'] = int(count > 100000)
    
    # PII 유형 (재추론 결과 사용)
    features['pii_type_inferred'] = row.get('pii_type_inferred', 'unknown')
    
    # 서버 정보 (가용 시)
    features['server_group'] = row.get('server_group', 'unknown')
    
    return features
```

**Rationale:**

- `inspect_count`는 분포가 극도로 편향(skewed)되어 있다. 단일 파일에서 233,498건이 검출되는 경우가 있으므로, log1p 변환 없이는 Boosting 모델의 분기점이 대부분 극단값에 집중되어 일반적인 케이스의 분류 성능이 떨어진다.
- `is_mass_detection`/`is_extreme_detection` 같은 이진 플래그는 Boosting 모델에서 "첫 번째 분기"로 사용되어 대량 검출 케이스를 빠르게 분리하는 역할을 한다. 이런 케이스는 거의 대부분 FP이므로, 이 단일 피처만으로도 상당한 볼륨의 분류가 가능하다.

**Missing Value 처리 원칙 — LightGBM Native NaN (Wave 1 구현):**

Dense 피처 결합 후 일괄 `fillna(0)` 처리는 **"값이 0"과 "값이 없음(NaN)"을 구분하지 못한다.** 예: `masking_ratio=0`은 "마스킹 없음"인지 "계산 불가"인지 알 수 없음.

**구현 원칙:** `fillna(0)` 제거. `pd.to_numeric(errors="coerce")`로 비수치 → NaN 변환은 유지. LightGBM/XGBoost는 NaN을 별도 분기로 처리하므로(`use_missing=True`, 기본값) native handling이 더 정확한 split을 학습한다.

```python
# 수정 전 (폐기)
dense = pd.concat(dense_parts, axis=1).fillna(0)
dense = dense.apply(pd.to_numeric, errors="coerce").fillna(0)

# 수정 후 - Wave 1: fillna(0) 제거
dense = pd.concat(dense_parts, axis=1)
dense = dense.apply(pd.to_numeric, errors="coerce")
# fillna(0) 제거 - LightGBM이 NaN을 native missing으로 처리

# 수정 후 - Wave 2 (2026-03-19): apply() → 컬럼별 루프 (OOM 방지)
# DataFrame.apply(fn)은 내부에서 중간 object array를 생성하므로
# 10M행 × 50컬럼 규모에서 메모리 스파이크가 발생한다.
# 컬럼별 in-place 할당으로 중간 배열 생성을 제거한다.
dense = pd.concat(dense_parts, axis=1)
for _col in dense.columns:
    dense[_col] = pd.to_numeric(dense[_col], errors="coerce")
# fillna(0) 없음 — LightGBM native NaN 처리 유지
```

> **구현 위치:** `src/features/pipeline.py` — dense 피처 결합 블록 (4개의 fillna(0) 제거, apply→컬럼 루프)

### 6.5 S2-5: 합성변수 3-Tier 정책 (v1.1 전면 재설계)

**v1.1 변경:** 전체 피처 대상 무차별 조합 확장은 과적합·메모리 폭발·설명가능성 훼손을 유발하며, Boosting 모델의 내재적 상호작용 학습과 중복된다. **기본값을 합성변수 OFF로 변경**하고, 3단계로 분리한다.

```
Tier 0 (기본값, 운영 배포): 합성변수 OFF
  → 수동 피처 + TF-IDF + path + tabular 그대로 (~1,055 피처)

Tier 1 (SAFE, 검증 후 운영 배포 가능): 선별적 합성변수 ON
  → 수동 이진/저카디널리티 피처 간 도메인 지식 기반 교차만
  → 10~20개 이내
  → min_support 필터 강제 (df ≥ 50)

Tier 2 (AGGRESSIVE, 연구/오프라인 분석 전용): 확장 합성변수
  → 운영 배포 금지, 실험 목적으로만 사용
```

**CLI 인터페이스:**

```bash
python run_training.py                        # 기본값: 합성변수 OFF (Tier 0)
python run_training.py --synth-tier safe      # Tier 1
python run_training.py --synth-tier aggressive # Tier 2 (연구용)
```

**Tier 1 SAFE 합성변수 후보 목록:**

| # | 합성변수 | 생성 규칙 | 도메인 근거 | evidence 매핑 |
|---|---------|----------|-----------|--------------|
| 1 | `log_file_AND_byte_kw` | `is_log_file × has_byte_kw` | 로그 파일의 bytes 패턴 = FP-bytes 강한 신호 | "로그 파일에서 bytes 키워드 발견" |
| 2 | `docker_AND_mass` | `is_docker_overlay × is_mass_detection` | Docker 환경 대량 검출 = 거의 확정 FP | "Docker overlay에서 10,000건+ 검출" |
| 3 | `email_AND_internal` | `(pii_type=email) × has_domain_kw` | 이메일 + 내부 도메인 키워드 | "이메일 패턴에서 내부 도메인 감지" |
| 4 | `mass_AND_system_path` | `is_mass_detection × has_system_token` | 시스템 경로 대량 검출 | "시스템 경로에서 대량 검출" |
| 5 | `timestamp_AND_digit_heavy` | `has_timestamp_kw × (digit_ratio > 0.6)` | 타임스탬프 키워드 + 숫자 비중 높음 | "타임스탬프 키워드와 높은 숫자 비율" |
| 6 | `license_path_AND_os_kw` | `has_license_path × has_os_copyright_kw` | 라이선스 경로 + OS 키워드 | "라이선스 경로에서 OS 키워드 발견" |
| 7 | `temp_path_AND_dev_kw` | `is_temp_or_dev × has_dev_kw` | 개발/테스트 경로 + 개발 키워드 | "개발 환경 경로에서 개발 키워드" |
| 8 | `extreme_AND_log` | `is_extreme_detection × is_log_file` | 10만건+ 로그 파일 = 거의 확정 FP | "로그 파일에서 100,000건+ 검출" |
| 9 | `digit_alpha_ratio` | `digit_ratio / (alpha_ratio + 0.01)` | 숫자 vs 문자 비율 관계 | "숫자 비중이 문자 대비 N배" |
| 10 | `email_AND_business_path` | `(pii_type=email) × has_business_token` | 이메일 + 업무 경로 = TP 방향 강화 | "업무 관련 경로에서 이메일 검출" |

**합성변수 채택 게이트 (Tier 1 → 운영 배포 조건):**

ablation 실험에서 **아래 4가지를 모두** 만족해야 채택한다.

1. TP Recall: 절대 감소 금지 (baseline 대비)
2. 고확신 FP precision: 유지 또는 상승
3. NEEDS_REVIEW 비율: baseline 대비 5%p 이상 증가 금지
4. ECE (calibration): 악화 금지

**평가는 반드시 pk_file Group Split + Time Split으로만 수행**한다. Event 랜덤 split에서 합성변수가 좋아 보이면 거의 확실히 누수다.

**Automated Ablation Pipeline (v1.2 추가):**

원칙 G(Zero-Human-in-the-Loop)에 따라, 합성변수 채택/탈락 결정을 사전 정의된 게이트로 자동 판정한다. 사람이 ablation 결과를 보고 결정할 필요가 없다.

```python
def automated_ablation(model_cls, X_base, y, synthetic_features: dict,
                       split_fn) -> dict:
    """
    각 합성변수를 자동으로 ablation 테스트하여 채택/탈락을 결정.

    Returns:
        adopted: 채택된 합성변수 목록
        rejected: 탈락된 합성변수 목록 + 탈락 사유
    """
    adopted, rejected = [], []
    train_idx, test_idx = split_fn(X_base, y)  # group_time_split

    # baseline (합성변수 없이)
    scores_without = evaluate(model_cls, X_base, y, train_idx, test_idx)

    for feat_name, feat_values in synthetic_features.items():
        X_with = np.hstack([X_base, feat_values.reshape(-1, 1)])
        scores_with = evaluate(model_cls, X_with, y, train_idx, test_idx)

        # 4가지 게이트 모두 통과해야 채택
        if (scores_with['tp_recall'] >= scores_without['tp_recall'] and
            scores_with['fp_precision'] >= scores_without['fp_precision'] and
            scores_with['review_rate'] <= scores_without['review_rate'] + 0.05 and
            scores_with['ece'] <= scores_without['ece']):
            adopted.append(feat_name)
        else:
            reasons = []
            if scores_with['tp_recall'] < scores_without['tp_recall']:
                reasons.append('tp_recall_drop')
            if scores_with['fp_precision'] < scores_without['fp_precision']:
                reasons.append('fp_precision_drop')
            if scores_with['review_rate'] > scores_without['review_rate'] + 0.05:
                reasons.append('review_rate_increase')
            if scores_with['ece'] > scores_without['ece']:
                reasons.append('ece_degradation')
            rejected.append({'feature': feat_name, 'reasons': reasons})

    return {'adopted': adopted, 'rejected': rejected}
```

**자동 판정 게이트 요약:**

| 게이트 | 조건 | 위반 시 |
|--------|------|---------|
| TP Recall | `scores_with.tp_recall >= scores_without.tp_recall` | 즉시 탈락 |
| FP Precision | `scores_with.fp_precision >= scores_without.fp_precision` | 즉시 탈락 |
| Review Rate | `scores_with.review_rate <= scores_without.review_rate + 0.05` | 즉시 탈락 |
| ECE (Calibration) | `scores_with.ece <= scores_without.ece` | 즉시 탈락 |

**리스크 근거 (무차별 합성변수 확장을 금지하는 이유):**

| 리스크 | 근거 |
|--------|------|
| **조합 폭발** | 전체 1,055개 피처 2차 확장 시 55만+ 차원, 128GB RAM 초과 가능 |
| **Sparse × Sparse 교차의 극단적 희소성** | TF-IDF 간 교차의 non-zero 비율 ≈ 0.15%, 과적합 확정적 |
| **Boosting과의 중복** | 트리 분기가 이미 피처 간 상호작용을 암묵적으로 학습 |
| **라벨 노이즈 증폭** | 합성변수가 노이즈 패턴을 더 강하게 학습 |
| **설명가능성 훼손** | TF-IDF 교차 피처의 SHAP 기여도는 사람이 납득 불가 |
| **누수 강화** | file_path 관련 합성변수가 "특정 파일 암기"를 촉진 |

### 6.6 S2-6: File-level Aggregation (선택)

```python
def compute_file_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    pk_file 단위로 검출 통계를 집계하여 개별 이벤트에 join.
    
    '10자 컨텍스트'의 한계를 구조적으로 보완하는 가장 강력한 방법.
    같은 파일에서 동일 패턴이 수만 번 반복되면 
    "업무 데이터 vs 로그/라이선스" 구분이 쉬워지는 경우가 많음.
    """
    agg = df.groupby('pk_file').agg(
        file_event_count=('pk_event', 'count'),
        file_unique_domains=('email_domain', 'nunique'),
        file_has_timestamp_kw_ratio=('has_timestamp_kw', 'mean'),
        file_has_bytes_kw_ratio=('has_byte_kw', 'mean'),
        file_pii_type_diversity=('pii_type_inferred', 'nunique'),
    ).reset_index()
    
    return agg
```

**Rationale:**

- 개별 이벤트(1행)만 보면 애매한 케이스도, 동일 파일의 전체 검출 패턴을 보면 판단이 쉬워지는 경우가 많다. "이 파일에서 500개의 이메일이 모두 @lguplus.co.kr 도메인" vs "이 파일에서 47개의 이메일이 모두 다른 도메인" — 전자는 내부 시스템 로그, 후자는 고객 데이터일 가능성이 높다.
- 이 집계는 원문이 없어도 가능한 **비식별적 맥락 증폭**이라 보안 제약에도 부합한다. 마스킹된 개인정보를 복원하지 않고도, "파일 수준의 통계적 특성"으로 분류 정보량을 늘릴 수 있다.

**⚠️ v1.1 누수 차단 주의사항:**

file-level aggregation이 train/test split 전에 전체 데이터로 계산되면 테스트 정보가 학습에 누수된다.

```python
# 수정 전 (누수 위험)
file_agg = compute_file_aggregates(df_all)  # 전체 데이터
df_all = df_all.merge(file_agg, on='pk_file')
X_train, X_test = split(df_all)

# 수정 후 (누수 차단)
train_idx, test_idx = group_time_split(df_all)
df_train = df_all.iloc[train_idx]
df_test = df_all.iloc[test_idx]

# train fold 내에서만 집계
file_agg_train = compute_file_aggregates(df_train)
df_train = df_train.merge(file_agg_train, on='pk_file', how='left')
df_test = df_test.merge(file_agg_train, on='pk_file', how='left')  # train 기준 적용
# test에만 있는 pk_file → NaN → fillna로 처리
```

### 6.7 출력 스키마

**`silver_features_base.parquet`**

S1의 모든 컬럼 + 추가 피처:

| 컬럼 그룹 | 대표 컬럼 | 생성 방식 |
|-----------|----------|-----------|
| 텍스트 정규화 | `raw_text`, `shape_text`, `path_text` | 문자 치환 / 엔트로피 축소 / 경로 토큰화 |
| 키워드 플래그 | `has_timestamp_kw`, `has_byte_kw`, `has_code_kw`, `has_domain_kw`, `has_os_copyright_kw`, `has_dev_kw`, `has_json_structure` | 키워드 리스트 매칭 |
| 구조/통계 피처 | `char_length`, `digit_ratio`, `special_char_ratio`, `alpha_ratio`, `has_at_sign`, `has_equals`, `has_colon`, `has_dot_separated`, `consecutive_digits_ratio`, `newline_count`, `masking_ratio` | 정규식/비율 계산 |
| 마스킹 패턴 피처 | `masked_digit_prefix_len`, `asterisk_count`, `masked_total_len`, `has_hyphen_in_masked` | 패턴 구조 분석 |
| 경로 피처 | `path_depth`, `is_log_file`, `is_docker_overlay`, `has_license_path`, `is_temp_or_dev`, `has_date_in_path`, `has_business_token`, `has_system_token` | 경로 토큰 매칭 |
| Tabular 피처 | `inspect_count_log1p`, `is_mass_detection`, `is_extreme_detection` | 변환/이진화 |
| Placeholder 비율 (Unknown-like 신호) | `placeholder_hash_ratio`, `placeholder_num_ratio` | placeholder 치환 빈도 → 모델이 "낯선 케이스" 감지하는 힌트 |

 > **[S3a 결합 후 추가 컬럼]** `rule_candidates_count`, `rule_has_conflict`는 S3a(RULE Labeler) 출력(§7.7)이며, ML 학습 시 S3a 실행 후 join되어 silver_features_base에 추가된다. S2 단계에서는 미포함.

**`silver_file_agg.parquet`** (선택)

pk_file 단위 집계 통계 → pk_event에 join하여 `silver_features_enriched.parquet` 생성.

### 6.8 Feature Selection — VarianceThreshold (Wave 2 구현)

Phase 2 full 모드(~8,055 피처)에서 TF-IDF 대부분은 noise다. Near-zero variance 피처를 제거하여 과적합 완화 및 학습 속도를 개선한다.

**구현 (`use_variance_threshold=True`):**

```python
from sklearn.feature_selection import VarianceThreshold

# sparse matrix에 직접 적용 가능
selector = VarianceThreshold(threshold=1e-5)
X_train_sel = selector.fit_transform(X_train)   # fit + transform
X_test_sel  = selector.transform(X_test)         # transform only

# feature_names 동기화
selected_mask  = selector.get_support()
feature_names_sel = [n for n, m in zip(feature_names, selected_mask) if m]
```

**적용 기준:**

| 모드 | 기본값 | 근거 |
|------|--------|------|
| Phase 1 (label-only, ~1,028 피처) | **OFF** | 피처 수가 적어 제거 효과 미미 |
| Phase 2 (full, ~8,055 피처) | **권장 ON** | TF-IDF 희귀/고빈도 noise 제거 효과 |

**후속 단계 (미착수):**
- LightGBM feature_importances_ 기반 top-K 선택 (학습 후 재학습)
- Permutation importance (통계적 유의성 검증)

> **구현 위치:** `src/features/pipeline.py` — VarianceThreshold 블록 (`use_variance_threshold` 파라미터)

### 6.9 대용량 OOM 방지 최적화 (2026-03-19, 10M행 서버 실운영 반영)

**배경:** 실서버 10,049,303행 × 126열 데이터 학습 중 OOM 크래시 발생.

```
numpy.core._exceptions._ArrayMemoryError:
Unable to allocate 520 GiB for an array with shape (83, 840941755)
```

원인은 row-wise Python 함수(`apply(fn, axis=1)`)가 10M행을 단일 Python 루프로 처리하면서 중간 dict/list 누적으로 메모리 스파이크를 유발한 것이다.

#### 6.9.1 build_meta_features() 벡터화 (`src/features/meta_features.py`)

`build_meta_features()` 내부의 3개 row-wise apply를 pandas/numpy 벡터화 연산으로 교체한다. 수치 결과는 완전히 동일하다.

| 이전 (row-wise) | 이후 (벡터화) | 효과 |
|----------------|--------------|------|
| `fname_col.apply(extract_fname_features)` | `str.contains()` 3개 컬럼 직접 할당 | 10~50배 속도, 메모리 스파이크 제거 |
| `result.apply(extract_detection_features, axis=1)` | `pd.to_numeric()` + `np.log1p()` + `pd.cut()` | axis=1 패턴 완전 제거 |
| `ts_col.apply(extract_datetime_features)` | `pd.to_datetime()` + `.dt` accessor | NaT → -1 처리 포함 |

```python
# 이전 (axis=1 row-wise — 10M행에서 치명적)
detect_feats = result.apply(extract_detection_features, axis=1)
detect_df = pd.DataFrame(list(detect_feats))

# 이후 (컬럼별 벡터화)
_cnt   = pd.to_numeric(result.get("pattern_count", _zero), errors="coerce").fillna(0.0)
_ssn   = pd.to_numeric(result.get("ssn_count",     _zero), errors="coerce").fillna(0.0)
_phone = pd.to_numeric(result.get("phone_count",   _zero), errors="coerce").fillna(0.0)
_email = pd.to_numeric(result.get("email_count",   _zero), errors="coerce").fillna(0.0)
result["pattern_count_log1p"]  = np.log1p(_cnt)
result["pattern_count_bin"]    = pd.cut(_cnt.clip(lower=0),
    bins=[0, 5, 20, 100, 1_000, np.inf], labels=[0,1,2,3,4],
    right=False, include_lowest=True).astype(float).fillna(0).astype(int)
result["is_mass_detection"]    = (_cnt > 10_000).astype(int)
result["is_extreme_detection"] = (_cnt > 100_000).astype(int)
result["pii_type_ratio"]       = _ssn / (_ssn + _phone + _email + 1)
```

> **주의:** 개별 함수(`extract_fname_features`, `extract_detection_features`, `extract_datetime_features`)는 backward 호환 및 단위 테스트용으로 유지. `build_meta_features()` 내부만 벡터화.

#### 6.9.2 Step 5 진입 전 컬럼 선택 (`scripts/run_training.py`)

`build_features(df, ...)` 호출 시 df 전체(150+열)를 넘기면 내부에서 `df_train / df_test` copy가 두 번 발생한다 (`df.loc[train_idx].reset_index()` 각 1회). 실제 피처 구성에 필요한 컬럼은 약 30개이므로, 진입 전에 슬라이싱하여 copy 크기를 최소화한다.

```python
# Step 4 완료 후, build_features 호출 직전
_KEEP_COLS = [
    "label_binary",
    "file_name", "file_path", "pk_file", "server_name",
    # meta_features 결과
    "fname_has_date", "fname_has_hash", "fname_has_rotation_num",
    "pattern_count_log1p", "pattern_count_bin",
    "is_mass_detection", "is_extreme_detection", "pii_type_ratio",
    "created_hour", "created_weekday", "is_weekend", "created_month",
    # path_features 결과
    "path_depth", "extension", "is_log_file", "is_docker_overlay",
    "has_license_path", "is_temp_or_dev", "is_system_device",
    "is_package_path", "has_cron_path", "has_date_in_path",
    "has_business_token", "has_system_token",
    # rule 컬럼
    "rule_matched", "rule_primary_class", "rule_id",
]
_keep = [c for c in _KEEP_COLS if c in df.columns]
df_for_features = df[_keep]   # 슬라이싱 (copy 없음)
result = build_features(df_for_features, label_column="label_binary", ...)
# df 자체는 유지 — Step 6b temporal split에서 label_work_month 필요
```

#### 6.9.3 예상 효과

| 최적화 항목 | 메모리 절감 | 속도 향상 |
|------------|-----------|---------|
| build_meta_features 벡터화 | ~30% (청크 내 스파이크 제거) | 10~50배 (axis=1 제거) |
| apply(pd.to_numeric) → 컬럼 루프 | ~5% | 2~3배 |
| Step 5 진입 전 컬럼 선택 | ~20% (df_train/df_test 크기) | 비례 |
| **합계** | **~45~55%** | **전체 파이프라인 ~1/3 단축** |

> **실운영 결과 (2026-03-20):** 10M행 데이터 OOM 없이 LightGBM 학습 완료. dense=31개 (meta/path 피처 22개 포함), F1-macro=0.9867 달성.

### 6.10 Detection Mode 피처 파이프라인 (Phase 1.5, 2026-03-19 구현)

`--source detection` 시 `silver_joined.parquet`(Sumologic + 레이블 JOIN 결과)에 포함된 `full_context_raw` 텍스트를 활용하여 multiview TF-IDF 피처를 생성한다.

| 뷰 | 소스 컬럼 | 설정 | 피처 수 |
|----|----------|------|--------|
| raw_text | full_context_raw | word, 1-2gram, max=5000 | 5,000 |
| shape_text | full_context_raw (문자 추상화) | char, 1-3gram, max=2000 | 2,000 |
| path_text | file_path (토큰화) | word, 1-2gram, max=1000 | 1,000 |
| dense | create_file_path_features + meta + path features | — | ~31 |
| **합계** | | | **~8,031** |

**아티팩트 저장 경로:**

| 파일 | Label 모드 | Detection 모드 |
|------|-----------|----------------|
| best_model | `models/final/best_model_v1.joblib` | `models/final/detection_best_model_v1.joblib` |
| feature_builder | `models/final/feature_builder.joblib` | `models/final/detection_feature_builder.joblib` |
| label_encoder | `models/final/label_encoder.joblib` | `models/final/detection_label_encoder.joblib` |
| ood_detector | `models/final/ood_detector.joblib` | `models/final/detection_ood_detector.joblib` |
| feature_schema | `models/final/feature_schema.json` | `models/final/detection_feature_schema.json` |
| feature_dir | `features/` | `features/detection/` |

**구현 위치:** `scripts/run_training.py` — `_run_detection_mode()` 함수

### 6.11 평가/리포트 피처 일관성 (2026-03-19 확정)

학습과 평가/리포트 시 동일한 피처 공간을 보장하기 위한 원칙:

**`run_evaluation.py`:** `X_test.npz` 직접 로드 → 학습과 100% 동일한 피처 행렬 사용.

**`run_poc_report.py`:** `FeatureBuilderSnapshot.transform(df)` 호출 → 학습 시 fitted TF-IDF vocab 그대로 사용(vocab 불일치 방지).

- Step 2 모델 경로: `rglob("*.joblib")` mtime 정렬 → `models/final/best_model_v1.joblib` 명시 경로 (calibrator/ood_detector 오로드 방지)
- Step 3 피처 준비: `build_meta_features()` → `extract_path_features()` → `FeatureBuilderSnapshot.transform()` 순서 보장

**Coverage-Precision 커브 정의:**
- `auto_fp_mask = P(FP) ≥ τ` — FP 클래스 확률만 사용 (top1_proba = max(P(TP), P(FP))는 TP 고신뢰 예측을 포함하여 precision 왜곡 발생)
- `precision = |{auto_fp ∩ 실제FP}| / |{auto_fp}|`
- `coverage = |{auto_fp ∩ 실제FP}| / |{전체 실제FP}|`

---
