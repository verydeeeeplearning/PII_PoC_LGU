## 3. 데이터 파이프라인 총괄

### 3.1 입력 데이터셋 구조

v1.2 학습/운영 기준 입력은 2개의 독립 데이터셋이며, PK(서버명+에이전트IP+파일경로) 기반으로 매핑된다.

**Dataset A — Server-i 검출 원본 데이터**

| 컬럼명 | 설명 | 샘플 값 |
|--------|------|---------|
| `dfile_guid` | 검출 건별 고유 ID (타임스탬프 포함) | `2025-12-04T09:05:06:000+0900` |
| `dfile_computername` | 서버/컴퓨터명 | (마스킹됨) |
| `dfile_username` | 사용자명 | (마스킹됨) |
| `dfile_agentip` | 에이전트 IP | `172.21.56.48` |
| `dfile_patternname` | 검출 패턴명 (PII 유형) | `핸드폰 번호`, `주민등록 번호` 등 |
| `dfile_inspectcount` | 검출 건수 | `116832`, `220498` 등 |
| `dfile_inspectedcontent` | 검출된 내용 (마스킹) | `010*****5639` |
| `dfile_inspectcontentwithcontext` | 검출 내용 + 앞뒤 약 10자 컨텍스트 | 아래 상세 |
| `dfile_filedirectedpath` | 파일 경로 (전체 경로) | `/JDINAS/R39/data1/...` |

**Dataset B — 레이블 Excel 데이터 (✅ 활용 확정)**

보안 담당자가 정탐/오탐을 수기로 판정한 결과물. 전 조직 2025년 3~12월(10개월) 확보.

**파일 구조 (72+개 파일):**
```
data/raw/label/
  25년 정탐 (3월~12월)/   ← label_raw = 정탐 강제 부여
    3월/
      정탐 취합 보고 자료_3월_CTO.xlsx
      정탐 취합 보고 자료_3월_NW부문.xlsx
      정탐 취합 보고 자료_3월_기업.xlsx
      ...
    4월/ ... 12월/
  25년 오탐 (3월~12월)/   ← label_raw = 오탐 강제 부여
    3월/ ... 12월/
```
- **`label_raw` 부여 규칙**: Excel 내 컬럼 값이 아니라 **폴더 위치 기준** 강제 변환
- **월별 데이터 수량 편차 큼** (수백 건 ~ 수만 건)
- **용량 초과 월**: CTO1/CTO2/CTO3 등 수동 분할 → PK 기반 중복 제거 필수
- **⚠️ 알려진 이슈**: `오탐 취합 보고 자료_7월_CTO` — 파일 열기 불가, 로딩 시 skip 처리

**컬럼 구조 (B~R, 17개 — 원본 컬럼명은 한글):**

| 위치 | 영문 변환명 | 역할 | 모델링 |
|------|-----------|------|--------|
| B | organization | 조직 | ❌ inference 시 미존재 |
| C | ops_dept | 운영/클렌징 부서 | ❌ |
| D | service | 서비스 | ❌ |
| E 🟡 | label_raw | 오탐여부(정탐/오탐) | **라벨** (파일 출처 기준 강제 부여) |
| F 🟡 | fp_description | 오탐 설명 / 정탐 삭제사유 | **다중 클래스 라벨 원천** (오탐 파일에만 존재) |
| G 🟡 | exception_requested | 예외요청 | ❌ |
| H 🟡 | retention_period | 파일 보관기간 | ❌ |
| I | server_name | 서버 이름 | ✅ 피처 |
| J | agent_ip | 에이전트 IP | ✅ 피처 |
| K | pattern_count | 패턴 개수 | ✅ 피처 |
| L | file_path | 파일 경로 | ✅ 핵심 피처 |
| M | file_name | 파일 이름 | ✅ 핵심 피처 |
| N | ssn_count | 주민등록번호 개수 | ✅ 피처 |
| O | phone_count | 핸드폰 번호 개수 | ✅ 피처 |
| P | email_count | E-Mail 주소 개수 | ✅ 피처 |
| Q | file_created_at | 파일 생성 일시 | ✅ 피처 |
| R | file_size | 파일 크기 | ❌ 피처 제외 (100% 결측) |

**모델링 피처 범위: I~Q (9개 컬럼, R 제외)** — B~H는 inference 시 미존재.

**⚠️ 한글 컬럼명 표기 불일치**: 조직/월별로 "서버 이름" vs "서버이름" vs "서버명" 등 미세한 차이 존재. 전처리 파이프라인에서 `config/column_name_mapping.yaml` 기반으로 영문 snake_case 변환 필수.

**레이블 Excel 물리 구조:**

실제 레이블 Excel 파일은 다음 형식을 따른다:

| 행 위치 | 내용 |
|---------|------|
| Row 1 (Excel 1행) | 타이틀 행 — `"오탐 취합 보고 자료_6월_CTO"` (데이터 아님, 파이프라인에서 스킵) |
| Row 2 (Excel 2행) | 컬럼 헤더 — `연번 \| 서버이름 \| 에이전트IP \| ...` |
| Row 3+ (Excel 3행~) | 데이터 — `1 \| srv01 \| 10.0.0.1 \| ...` (연번은 파이프라인에서 자동 제거) |

`ingestion_config.yaml`의 `excel_read` 섹션에서 설정 가능 (`header_row: 1`, `skip_first_col: true`).

**⚠️ 핵심 부재 피처**: `dfile_inspectcontentwithcontext`(마스킹 텍스트 컨텍스트)가 레이블 데이터에 없음. 이는 아키텍처 설계의 핵심 입력으로, Sumologic JOIN 없이는 사용 불가. → Phase 1은 파일 메타데이터 기반 분류, Phase 2(JOIN 후)에서 텍스트 피처 활성화.

**Dataset C — 현업 피드백 데이터 (❌ v1.2 범위 제외)**

현업 담당자가 실서버 확인 후 정탐/오탐을 자유 텍스트로 기입한 결과. 비정형 사유를 클래스로 구조화하는 공수 대비 효익이 낮아 v1.2 학습/운영 파이프라인에서는 사용하지 않는다.

**Mock 데이터 생성 (`scripts/generate_mock_raw_data.py`):**

실제 데이터 투입 전 파이프라인 검증을 위해 원본 포맷 mock 데이터를 생성하는 스크립트. 레이블 Excel과 Sumologic Excel(.xlsx)을 동시에 생성한다. `--csv` 옵션으로 CSV도 병행 출력 가능.

```bash
python scripts/generate_mock_raw_data.py                              # 기본: 200행, Excel 출력
python scripts/generate_mock_raw_data.py --label-rows 500 --sumologic-rows 500
python scripts/generate_mock_raw_data.py --csv                        # Excel + CSV 병행 출력
```

생성 구조:
- `data/raw/label/25년 정탐 (3월~12월)/월/정탐 취합 보고 자료_월_조직.xlsx`
- `data/raw/label/25년 오탐 (3월~12월)/월/오탐 취합 보고 자료_월_조직.xlsx`
- `data/raw/dataset_a/sumologic_mock_202506_202507.xlsx`  (기본, `--csv` 옵션 시 `.csv`도 추가 생성)

특징: 한글 컬럼명 variant 2종 교대 사용(표기 불일치 재현), FP 7가지 시나리오(로그/docker/OSS저작권/더미테스트/내부도메인/타임스탬프/bytes), Sumologic에 JOIN 키 컬럼(`dfile_filedirectedpath`, `dfile_filename`, `dfile_filecreatedtime`) 포함, 실제 Excel 구조 재현(Row 1=타이틀, Row 2=연번 포함 헤더, Row 3+=데이터 — `ingestion_config.yaml excel_read` 설정과 정합).

### 3.2 마스킹 현황 & 모델링 핵심 제약

| 항목 | 상태 |
|------|------|
| Server-i 자체 마스킹 | ✅ 검출 시점에 애스터리스크 처리 |
| Splunk 전달 시점 | 이미 마스킹 완료 |
| 원본 데이터 확인 | ❌ 불가 — 마스킹된 데이터만 활용 가능 |
| 모델 입력 | 마스킹된 컨텍스트 약 10자 + Tabular 메타데이터 |

**모델링 핵심 제약:** 모델도 마스킹된 컨텍스트 약 10자로만 판단해야 한다. 이는 소만사 운영 인력의 판단 조건과 동일하다. 다만, 이메일의 경우 로컬 파트만 마스킹되고 도메인은 노출되는 구조(`****@bdp.lguplus.co.kr`)이므로, 도메인 기반 판단은 가능하다.

### 3.3 PK 기반 원본 매핑 전략

TF-IDF, Bag-of-Words 등의 텍스트 변환은 역변환이 불가능하므로, 모델 추론 결과를 운영 프로세스(담당자 통보, 공유폴더 전달, 이메일 공지)에 활용하기 위해서는 PK 기반 원본 매핑이 필수적이다.

**PK 설계 (확정, 2026-02-26):**

| PK | 구성 | 용도 |
|----|------|------|
| `pk_event` | `SHA256(server_name\|agent_ip\|file_path\|file_name\|file_created_at)` | 이벤트 수준 식별 · 라벨 충돌 감지 · Sumologic JOIN 키 |
| `pk_file` | `SHA256(server_name\|agent_ip\|file_path\|file_name)` | 파일 수준 집계 · GroupShuffleSplit 기준 · 파일 레벨 누수 차단 |

**Dataset A(Sumologic) 필드 매핑:**

| PK 구성 요소 | Dataset A 컬럼명 | Dataset B 컬럼명(영문 변환 후) |
|-------------|-----------------|-------------------------------|
| 서버명 | `dfile_computername` | `server_name` |
| 에이전트 IP | `dfile_agentip` | `agent_ip` |
| 파일 경로 | `dfile_filedirectedpath` | `file_path` |
| 파일 이름 | `dfile_filename` | `file_name` |
| 파일 생성 시점 | `dfile_filecreatedtime` | `file_created_at` |

**⚠️ Sumologic JOIN 주의사항:**
- 기존 추출분에서 `dfile_filedirectedpath`, `dfile_filename`, `dfile_filecreatedtime` **누락** → 재추출 필요
- 레이블은 작업 시점 기준, Sumologic은 주차별 스냅샷 기준 → 시점 불일치 가능
- 주차별 스냅샷 append 구조로 인해 레이블 1건 : Sumologic 多건 매핑 가능 (1:多 처리 전략 필요)

**파이프라인 내 PK 흐름:**
```
원본 데이터 → PK 생성(S1) → Feature Engineering(S2)
                │                        │
                ▼                        ▼
         원본 저장소에 보관         변환된 피처로 추론
                │                        │
                └────── PK 기반 JOIN ─────┘
                              │
                              ▼
                    원본 + 예측 결과 통합
```

**왜 역변환이 아니라 PK 기반 매핑인가:**

| 변환 유형 | 역변환 가능 여부 | 문제점 |
|-----------|-----------------|--------|
| StandardScaler | ✅ `inverse_transform()` | 부동소수점 오차, Scaler 객체 유지 필요 |
| TF-IDF, Bag-of-Words | ❌ 역변환 무의미 | 원본 텍스트 복원 불가 |
| 파생 변수 (경로 depth 등) | ❌ 불가 | 정보 손실, 원본 복원 불가 |

PK 기반 접근은 어떤 변환을 적용해도 PK만 유지하면 원본 복원이 보장되므로, Feature Engineering의 복잡도와 무관하게 안정적이다.

---

## 4. Stage S0: Raw Ingest & Storage

### 4.1 기능

Dataset A(Excel/.xlsx 기본, CSV도 허용), Dataset B(Excel) 원본을 그대로 보관하고, Schema Canonicalization(스키마 정규화)을 수행한다.

**수행 작업:**

1. **원본 보관**: 어떠한 내용 변환도 수행하지 않는 원본 백업
2. **인코딩/줄바꿈 정리**: CSV 멀티라인 필드 깨짐 방지(quote/escape 처리) + 인코딩 자동 감지/고정(utf-8-sig, cp949 등)
3. **컬럼명 통일**: 원본 컬럼명을 정규 스키마로 매핑
   - `dfile_computername` → `server_name`
   - `dfile_agentip` → `agent_ip`
   - `dfile_filedirectedpath` → `file_path`
   - `dfile_filename` → `file_name`
   - `dfile_patternname` → `pattern_name_raw`
   - `dfile_inspectcount` → `inspect_count`
   - `dfile_inspectedcontent` → `masked_hit`
   - `dfile_inspectcontentwithcontext` → `full_context_raw`
4. **Bronze Parquet 생성**: `bronze_events.parquet`, `bronze_labels.parquet`

**Rationale (Schema Canonicalization):**

- 패턴이 늘어나도 **입력 포맷 흔들림(인코딩/줄바꿈/컬럼명)**은 항상 발생할 수 있다 → 먼저 스키마를 고정해야 뒤 단계가 안정적이다.
- "원문 보존"을 위해 Bronze 단계는 가공 최소화한다.

### 4.2 출력

```
raw/
├── yyyymm/
│   ├── dataset_a/
│   │   └── 12월_거래주_검출내역.csv
│   ├── dataset_b/
│   │   └── 오탐case정리_260121.xlsx
```

### 4.2b Silver Parquet 산출물 (processed/)

S0–S1 파이프라인이 생성하는 세 가지 Silver Parquet 산출물:

| 파일 | 생성 명령 | 설명 |
|------|----------|------|
| `silver_label.parquet` | `--source label` | 레이블 Excel → pk_file/pk_event SHA256 포함 정규화 |
| `silver_detections.parquet` | `--source detection` | Sumologic Excel → dfile_* 컬럼 매핑 + pk_file SHA256 계산 (DataSourceRegistry가 column_map/pk_fields 관리, `--datasource` 옵션으로 소스 선택) |
| `silver_joined.parquet` | `--source joined` | silver_label + silver_detections → pk_file 기준 inner JOIN |

#### silver_joined.parquet (full 모드 전용)

silver_label.parquet과 silver_detections.parquet을 pk_file 기준으로 inner join한 결과.
레이블 데이터(정탐/오탐 ground truth)와 Sumologic 검출 데이터를 연결하여 ML 학습에 사용한다.

| 컬럼 | 출처 | 설명 |
|------|------|------|
| `pk_file` | 공통 JOIN 키 | SHA256(server_name\|agent_ip\|file_path\|file_name) |
| `label_raw` | silver_label | TP/FP ground truth |
| `full_context_raw` | silver_detections | Sumologic 검출 원문 컨텍스트 |
| 나머지 컬럼 | 양쪽 테이블 상속 | label/detection 원본 컬럼 모두 포함 |

**의존성:** silver_label.parquet, silver_detections.parquet 모두 먼저 생성 필요
**주의:** JOIN은 동일한 `generate_mock_raw_data.py` 실행으로 생성된 데이터일 때만 pk_file 일치 보장

### 4.3 Schema Registry (v1.1 추가)

입력 데이터의 스키마가 변경되었을 때 파이프라인이 crash하지 않고, 명시적으로 격리(quarantine)하도록 Schema Registry를 도입한다.

```yaml
# schema_registry.yaml
dataset_a:
  version: "2026-02"
  required_columns:
    - dfile_computername
    - dfile_agentip
    - dfile_filedirectedpath
    - dfile_inspectedcontent
    - dfile_inspectcontentwithcontext
  optional_columns:
    - dfile_username
  rename_map:
    dfile_computername: server_name
    dfile_agentip: agent_ip
    # ...
  on_missing_required: "quarantine"  # crash가 아니라 격리
  on_unknown_column: "ignore_and_log"
```

**Rationale:** 원칙 F(Open-World Defense)에 따라, 스키마 불일치 시 파이프라인이 중단되지 않고 해당 행을 격리한 뒤 알림을 생성한다. 이를 통해 새로운 데이터 포맷이 유입되어도 기존 처리 흐름은 유지된다.

### 4.4 Auto-Schema-Detector (v1.2 추가)

원칙 G(Zero-Human-in-the-Loop)에 따라, 데이터 포맷 변경 시 사람이 `schema_registry.yaml`을 수동 수정하는 대신, 시스템이 자동으로 스키마를 감지하고 매핑을 생성한다.

```python
from difflib import SequenceMatcher

class AutoSchemaDetector:
    """
    새 데이터의 컬럼명을 기존 schema_registry와 퍼지 매칭하여
    자동으로 rename_map을 생성/갱신한다.
    """
    FUZZY_THRESHOLD = 0.70  # 유사도 임계값

    def detect_and_update(self, new_columns: list[str],
                          schema_registry: dict) -> dict:
        """
        Returns:
            updated_schema: 갱신된 schema_registry
            report: 매핑 결과 리포트
        """
        existing_map = schema_registry.get('rename_map', {})
        known_sources = set(existing_map.keys())
        known_targets = set(existing_map.values())

        report = {'auto_mapped': [], 'unmapped_required': [],
                  'unmapped_optional': [], 'already_known': []}

        for col in new_columns:
            if col in known_sources:
                report['already_known'].append(col)
                continue

            # 기존 rename_map의 소스 컬럼명과 퍼지 매칭
            best_match, best_score = None, 0.0
            for existing_src, target in existing_map.items():
                score = SequenceMatcher(None, col.lower(),
                                       existing_src.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_match = (existing_src, target)

            if best_score >= self.FUZZY_THRESHOLD and best_match:
                # 자동 매핑 성공
                existing_map[col] = best_match[1]
                report['auto_mapped'].append({
                    'new_col': col,
                    'matched_to': best_match[0],
                    'target': best_match[1],
                    'similarity': round(best_score, 3)
                })
            else:
                # 매핑 실패 → required/optional에 따라 처리
                if col in schema_registry.get('required_columns', []):
                    report['unmapped_required'].append(col)
                else:
                    report['unmapped_optional'].append(col)

        schema_registry['rename_map'] = existing_map
        schema_registry['version'] = f"auto_{datetime.now():%Y%m%d}"

        return schema_registry, report
```

**동작 흐름:**

1. 새 데이터 도착 시 컬럼명 목록 자동 추출
2. 기존 `schema_registry.yaml`의 `rename_map`과 퍼지 매칭 (유사도 ≥ 0.70)
   - `dfile_computername` ↔ `computer_name` → 유사도 0.85 → 자동 매핑
3. 매핑 실패한 컬럼:
   - required → quarantine (기존 정책 유지)
   - optional → `ignore_and_log` (기존 정책 유지)
4. 성공적으로 매핑된 새 스키마 → `schema_registry.yaml` 자동 갱신 (Git auto-commit)
5. 다음 실행부터 새 스키마 자동 적용

### 4.5 전처리 설정 외부화 (`preprocessing_config.yaml`, v1.3 추가)

하드코딩 상수를 `config/preprocessing_config.yaml`로 외부화하여, 데이터 포맷 변경 시 코드 수정 없이 YAML만 수정하면 대응 가능하게 한다.

**주요 설정 섹션:**

| 섹션 | 내용 |
|------|------|
| `data_sources.sumologic_server_i.column_map` | `dfile_*` → 표준 컬럼명 매핑 (15개) |
| `data_sources.sumologic_server_i.pk_file_fields` | pk_file 생성에 사용할 필드 목록 |
| `join.label_cols` | JOIN 시 silver_label에서 가져올 컬럼 목록 |
| `join.on_zero_join` | JOIN 0건 시 정책 (`warn_and_diagnose`: 자동 진단 출력) |
| `parsing.s1_parser` | `window_size`, `masking_pattern`, `pattern_truncation`, `context_truncation` |
| `parsing.label_loader` | `datetime_formats`, `on_no_valid_sheet`, `on_datetime_parse_fail` |
| `parsing.loader` | `csv_encoding_candidates` |

### 4.6 DataSourceRegistry (`src/data/datasource_registry.py`, v1.3 추가)

`DataSourceRegistry`는 `preprocessing_config.yaml`의 `data_sources` 섹션을 읽어, 새 검출 데이터 소스를 YAML에만 추가하면 코드 변경 없이 즉시 인식하도록 한다.

```bash
# --datasource 옵션으로 소스 선택 (default: sumologic_server_i)
python scripts/run_data_pipeline.py --source detection --datasource sumologic_server_i
```

**API:**
- `get_column_map(source_name)` — 컬럼 매핑 dict 반환
- `get_pk_fields(source_name)` — pk_file 생성 필드 목록 반환
- `find_files(source_name)` — file_pattern glob으로 파일 탐색
- `list_sources()` — 등록된 소스 이름 목록 반환

**새 소스 추가 방법**: `preprocessing_config.yaml`의 `data_sources`에 새 블록 추가만으로 즉시 `--datasource <new_source>` 사용 가능.

### 4.7 에러 처리 정책 (v1.3 추가)

#### 4.7.1 Excel 시트 폴백 정책

`LabelLoader._read_excel_sheets()`는 `on_no_valid_sheet` 설정에 따라 유효 시트 없는 경우를 처리한다:

| 정책값 | 동작 |
|--------|------|
| `"warn_and_first_fallback"` (기본) | 경고 로그 출력 + 첫 번째 시트를 fallback으로 사용 |
| `"raise"` | `ValueError` 발생 (엄격한 운영 환경) |
| `"skip"` | 해당 파일 전체 건너뜀 |

#### 4.7.2 datetime 파싱 실패 격리 (Quarantine)

`on_datetime_parse_fail: "quarantine"` 설정 시:
- `file_created_at` 파싱 실패 행 → `data/processed/silver_quarantine.parquet` 격리 저장 (`quarantine_reason = "datetime_parse_fail"`)
- 정상 파싱 행만 silver_label.parquet에 포함
- `"warn"` 설정 시: 기존 동작 유지 (경고만, 행은 NaT로 유지)

#### 4.7.3 JOIN 0건 자동 진단

`join.on_zero_join: "warn_and_diagnose"` 설정 시 JOIN 결과 0건이면 `_diagnose_join_mismatch()` 자동 호출:
- pk_file 길이 검증 (기댓값 64자, SHA256)
- silver_label/silver_detections 각각의 pk_file 샘플 출력
- 불일치 원인 추정 메시지 출력

### 4.8 컬럼명 Robust 처리 (v1.3 추가)

#### 4.8.1 컬럼명 공백/탭 자동 strip

`ColumnNormalizer.normalize()`의 첫 단계에서 모든 컬럼명 앞뒤 공백/탭을 자동으로 제거한다:

| 원본 컬럼명 | strip 후 | 최종 변환 |
|------------|---------|----------|
| `" 서버이름"` | `"서버이름"` | `server_name` |
| `"서버이름 "` | `"서버이름"` | `server_name` |
| `"\t서버이름"` | `"서버이름"` | `server_name` |

`_is_valid_sheet()`도 동일하게 stripped set으로 비교하여 공백이 있는 컬럼명도 유효 시트로 정상 인식한다.

#### 4.8.2 컬럼명 처리 범위 요약

| 케이스 | 처리 방식 | 설정 |
|--------|----------|------|
| 컬럼 순서 다름 | pandas 헤더명 기반 읽기 → 이미 robust | — |
| 앞뒤 공백/탭 | `normalize()` 첫 단계 자동 strip | — |
| 다른 한글 표기 (`서버명`, `서버 이름`) | YAML aliases | `config/column_name_mapping.yaml` |
| 미등록 한글 컬럼 (기본) | 경고 로그 출력 | `strict=False` (기본) |
| 미등록 한글 컬럼 (엄격) | `ValueError` 발생 | `ColumnNormalizer(strict=True)` |

### 4.9 Excel 물리 구조 설정화 (v1.3 추가)

레이블 Excel의 행 구조가 조직/월별로 다를 수 있어, `ingestion_config.yaml`의 `excel_read` 섹션으로 설정화한다:

```yaml
excel_read:
  header_row: 1       # 0-indexed. 1 = Excel 2행이 헤더 (Excel 1행은 타이틀 행으로 스킵)
  skip_first_col: true # true = 첫 번째 컬럼 제거 (연번/순번 등 일련번호)
```

`LabelLoader._read_excel_sheets()` 동작:
1. `pd.read_excel(header=self._header_row)` — 헤더 행 위치 적용
2. `skip_first_col=True`이면 `df.iloc[:, 1:]` — 첫 컬럼 제거
3. CSV 파일에는 미적용 (CSV는 `header_row`/`skip_first_col` 무시)

### 4.10 Rationale

**"원본 그대로" 보관은 감사/재현성의 시작점이다.**

- 룰/모델이 바뀌었을 때 "그때 원본이 뭐였지?"를 확인할 수 없으면 운영 신뢰가 무너진다. 개선 사이클의 전제 조건은 "동일 원본으로 재실행하여 비교할 수 있는 것"이다.
- 파싱/정규화(S1) 과정에서 정보 손실이 발생할 수 있다(인코딩 이슈, 멀티라인 파싱 실패 등). 원본이 보존되어 있으면 S1 파서를 개선한 뒤 재파싱이 가능하다.
- 보안 감사(Audit) 시 "이 예측의 근거가 된 원본 데이터"를 추적해야 할 때, 원본 보관이 되어 있어야 end-to-end 추적이 가능하다.
- 스토리지 비용은 무시할 수 있다(월 100~150만 건 CSV는 수 GB 수준, 3.6TB RAID 1 기준 충분).

---

## 5. Stage S1: Normalize & Parse

### 5.1 기능

이 파이프라인에서 **가장 중요한 단계**다. 학습/추론의 기본 단위(1행=1검출 이벤트)를 결정하고, 이후 모든 처리의 기반이 된다.

**수행 작업:**

1. **인코딩/줄바꿈/구분자 정리**: UTF-8 정규화(NFKC), Windows/Unix 줄바꿈 통일(`\r\n`→`\n`), 탭→공백, 과도한 공백 축소
2. **컬럼명 통일 (Schema Canonicalization)**: 원본 컬럼명을 정규 스키마로 매핑 (예: `dfile_computername` → `server_name`, `dfile_inspectedcontent` → `masked_hit`, `dfile_inspectcontentwithcontext` → `full_context_raw`)
3. **"1셀=다중검출" → "1행=1검출 이벤트" 분해**: `full_context_raw` 필드에서 마스킹 패턴 중심으로 이벤트를 분리
4. **컨텍스트 분리**: `left_ctx` / `masked_pattern` / `right_ctx` / `full_context` 생성
5. **파일 경로 정규화**: 소문자화, 구분자 통일(`\`→`/`), 중복 슬래시 정리, `path_depth`/`extension` 등 파생 컬럼 생성
6. **full_context 정규화 (비파괴)**: 유니코드 정규화(NFKC), 줄바꿈 통일, 과도한 공백 축소 → `full_context_norm` 파생 컬럼으로 생성 (원문 `full_context_raw` 보존)
7. **앵커 기반 local_context 생성**: `masked_hit`가 `full_context_raw` 안에 존재하면 해당 위치 기준 ±W(60~120 chars) 윈도우를 `local_context_raw`로 생성. 못 찾으면 앞부분/가운데를 최대 N chars(200~400)로 절단
8. **PK 생성**: `pk_file` = SHA256(server_name|agent_ip|file_path|file_name), `pk_event` = SHA256(server_name|agent_ip|file_path|file_name|file_created_at)
9. **PII 유형 재추론**: `pii_type_inferred` 생성 (패턴 종류 필드 신뢰도 이슈 대응)
10. **이메일 도메인 추출**: `@` 이후 문자열을 `email_domain` 필드로 분리

### 5.2 "1행=1검출 이벤트" 파싱 규칙 — 3단 폴백 구조 (v1.1 개선)

`dfile_inspectcontentwithcontext`는 여러 줄/다중 검출이 섞일 수 있는 복잡한 필드다.

**v1.1 변경:** 마스킹 방식 변경(예: `X`, `#`, `[MASKED]`, `*` 3개 미만) 시 이벤트가 0건 생성되어 데이터가 누락되는 문제를 방지하기 위해, 3단 폴백 구조로 변경한다.

```
1차 시도: 현 방식 (마스킹 패턴 *{3,} 기반 분해)
    ↓ 실패 시
2차 시도: masked_hit를 앵커로 full_context에서 위치 탐색
          → 찾으면 주변 ±W chars를 이벤트로 생성
    ↓ 실패 시
3차 폴백: 해당 row 전체를 1 event로 생성
          → parse_status = "FALLBACK_SINGLE_EVENT"
```

**권장 파싱 접근:**

```python
import re
import hashlib

def parse_context_field(
    raw_text: Optional[str],
    pk_file: str,
    masked_hit: Optional[str] = None,
    window: int = 60,
    config: Optional[S1ParserConfig] = None,  # v1.3 추가: None이면 S1ParserConfig.defaults() 사용
) -> list[dict]:
    """
    하나의 컨텍스트 셀에서 개별 검출 건을 분리.

    파싱 전략 (3단 폴백):
    1차: 마스킹 패턴(*{3,}) 중심으로 좌/우 컨텍스트를 잘라냄
    2차: masked_hit 앵커 기반 위치 탐색
    3차: row 전체를 단일 이벤트로 생성

    완벽한 파싱이 아니라 '일관된 파싱'이 핵심.
    학습/추론이 동일 파서를 사용하면 모델이 그 일관성을 학습한다.
    """
    if not raw_text or pd.isna(raw_text):
        return []

    # === 1차 시도: 기존 마스킹 패턴 기반 ===
    results = _parse_by_masking_pattern(raw_text, pk_file)
    if results:
        for r in results:
            r['parse_status'] = 'OK'
        return results

    # === 2차 시도: masked_hit 앵커 기반 ===
    if masked_hit:
        results = _parse_by_anchor(raw_text, pk_file, masked_hit)
        if results:
            for r in results:
                r['parse_status'] = 'FALLBACK_ANCHOR'
            return results

    # === 3차 폴백: 전체를 단일 이벤트로 ===
    pk_event = hashlib.sha256(f"{pk_file}_0".encode()).hexdigest()
    return [{
        'pk_event': pk_event,
        'pk_file': pk_file,
        'event_index': 0,
        'left_ctx': '',
        'masked_pattern': raw_text[:50],
        'right_ctx': '',
        'full_context': raw_text[:200],
        'parse_status': 'FALLBACK_SINGLE_EVENT',
    }]


def _parse_by_masking_pattern(raw_text: str, pk_file: str) -> list[dict]:
    """1차: *{3,} 패턴 기반 파싱"""
    lines = raw_text.split('\n')
    results = []
    event_index = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        pattern = r'(.{0,15})(\S*\*{3,}\S*)(.{0,15})'
        matches = list(re.finditer(pattern, line))

        for m in matches:
            left_ctx = m.group(1).strip()
            masked_pattern = m.group(2)
            right_ctx = m.group(3).strip()
            full_context = f"{left_ctx} {masked_pattern} {right_ctx}".strip()

            pk_event = hashlib.sha256(f"{pk_file}_{event_index}".encode()).hexdigest()

            results.append({
                'pk_event': pk_event,
                'pk_file': pk_file,
                'event_index': event_index,
                'left_ctx': left_ctx,
                'masked_pattern': masked_pattern,
                'right_ctx': right_ctx,
                'full_context': full_context,
            })
            event_index += 1

    return results


def _parse_by_anchor(raw_text: str, pk_file: str, masked_hit: str, window: int = 60) -> list[dict]:
    """2차: masked_hit 앵커 기반 위치 탐색"""
    idx = raw_text.find(masked_hit)
    if idx == -1:
        return []

    start = max(0, idx - window)
    end = min(len(raw_text), idx + len(masked_hit) + window)
    context = raw_text[start:end]

    pk_event = hashlib.sha256(f"{pk_file}_0".encode()).hexdigest()
    return [{
        'pk_event': pk_event,
        'pk_file': pk_file,
        'event_index': 0,
        'left_ctx': raw_text[start:idx].strip(),
        'masked_pattern': masked_hit,
        'right_ctx': raw_text[idx+len(masked_hit):end].strip(),
        'full_context': context,
    }]
```

### 5.2b S1ParserConfig (v1.3 추가)

파서 파라미터를 dataclass로 관리하여 YAML 기반 설정과 코드 기반 테스트를 모두 지원한다.

```python
@dataclass
class S1ParserConfig:
    window_size: int = 60                                    # 앵커 기반 컨텍스트 윈도우
    masking_pattern: str = r'(.{0,15})(\S*\*{3,}\S*)(.{0,15})'  # 1차 파싱 정규식
    pattern_truncation: int = 50                             # 3차 폴백 masked_pattern 길이 제한
    context_truncation: int = 200                            # 3차 폴백 full_context 길이 제한

    @classmethod
    def from_yaml(cls, config_path=None) -> "S1ParserConfig":
        """preprocessing_config.yaml의 parsing.s1_parser 섹션 로드. 실패 시 defaults."""
        ...
```

파라미터 기본값은 `preprocessing_config.yaml`의 `parsing.s1_parser` 섹션과 동일하게 유지되며, `parse_context_field(config=None)` 호출 시 기존 동작과 완전히 동일하다.

### 5.3 PII 유형 재추론

패턴 종류 필드(`dfile_patternname`)의 신뢰도가 낮으므로, 검출 내역에서 PII 유형을 독립적으로 재추론한다.

```python
def infer_pii_type(masked_pattern: str, full_context: str) -> str:
    """
    패턴 종류 필드의 신뢰도 이슈 대응.
    검출 내역의 형태적 특성으로 PII 유형을 재추론.
    """
    if '@' in full_context:
        return 'email'
    elif re.search(r'01[016789][\-\*]', full_context):
        return 'phone'
    elif re.search(r'\d{6}[\-\*]\d?\*', masked_pattern):
        return 'rrn'  # 주민등록번호
    else:
        return 'unknown'
```

### 5.4 Quarantine 테이블 & 파싱 KPI (v1.1 추가)

`silver_detections.parquet`과 병행하여 `silver_quarantine.parquet`를 생성한다.

| 조건 | 처리 |
|------|------|
| 파싱 성공 | `silver_detections.parquet`에 기록, `parse_status="OK"` |
| 폴백으로 성공 | `silver_detections.parquet`에 기록, `parse_status="FALLBACK_*"` |
| 스키마 불일치/필수 필드 누락 | `silver_quarantine.parquet`에 격리, `quarantine_reason` 기록 |
| 인코딩 오류 | `silver_quarantine.parquet`에 격리 |

**필수 KPI:**

```
row_parse_success_rate = (#파싱 실패 없이 처리된 원본 row) / (#원본 row)
avg_events_per_row = (#생성 pk_event) / (#원본 row)   # 1:多 분해 비율, 1.0 초과 가능
fallback_rate = (#폴백 이벤트) / (#전체 이벤트)
quarantine_count = 격리된 행 수
```

`row_parse_success_rate`가 95% 미만으로 떨어지면 경보를 발생시킨다. 이는 S1이 새로운 데이터 포맷에 대응하지 못하고 있다는 가장 빠른 신호다.

### 5.5 출력 스키마

**`silver_detections.parquet`** (1행=1검출 이벤트)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `pk_event` | string | 검출 이벤트 고유 ID (해시) |
| `pk_file` | string | 파일 단위 고유 ID (해시) |
| `event_index` | int | 동일 파일 내 이벤트 순번 |
| `server_name` | string | 서버/호스트명 |
| `agent_ip` | string | 에이전트 IP |
| `file_path` | string | 파일 전체 경로 (정규화: 소문자, `/` 통일) |
| `file_name` | string | 파일명 |
| `extension` | string | 파일 확장자 |
| `path_depth` | int | 경로 깊이 (`/` 개수) |
| `inspect_count` | int | 파일 내 총 검출 건수 |
| `masked_hit` | string | 마스킹된 검출값 원문 (= `inspectedmaskedcontent`) |
| `left_ctx` | string | 마스킹 패턴 좌측 컨텍스트 |
| `masked_pattern` | string | 마스킹된 검출 패턴 |
| `right_ctx` | string | 마스킹 패턴 우측 컨텍스트 |
| `full_context_raw` | string | 원문 컨텍스트 (비파괴 보존) |
| `full_context_norm` | string | 정규화된 컨텍스트 (NFKC, 줄바꿈/공백 통일) |
| `full_context` | string | 좌+마스킹+우 결합 문자열 |
| `local_context_raw` | string | 앵커 기반 윈도우링된 컨텍스트 (masked_hit 기준 ±W chars) |
| `pattern_name_raw` | string | Server-i 원본 패턴 종류 (참고용, dfile_patternname → §4.1) |
| `pii_type_inferred` | string | 재추론된 PII 유형 (email/phone/rrn/unknown) |
| `email_domain` | string (nullable) | 이메일 도메인 (이메일인 경우만) |
| `detection_time` | datetime | 검출 시점 |
| `label` | string (nullable) | 학습 데이터인 경우 정답 라벨 |
| `parse_status` | string | 파싱 상태 (`OK`, `FALLBACK_ANCHOR`, `FALLBACK_SINGLE_EVENT`) |

**`silver_quarantine.parquet`** (v1.1 추가 — 파싱 실패/격리 행)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `row_id` | int | 원본 행 번호 |
| `quarantine_reason` | string | 격리 사유 (`SCHEMA_MISMATCH`, `ENCODING_ERROR`, `MISSING_REQUIRED_FIELD`) |
| `raw_content` | string | 원본 행 내용 (디버깅용) |
| `quarantine_time` | datetime | 격리 시점 |

### 5.6 Rationale

**학습/추론의 기본 단위가 흔들리면 모델이 무너진다.**

- "1행에 여러 케이스가 섞인 텍스트"는 라벨과의 관계가 흐려지고, 모델은 결국 잡음(noise)을 학습하게 된다. 1행=1검출 이벤트로 정규화해야 "이 텍스트 → 이 라벨" 관계가 명확해진다.
- `inspectcontentwithcontext` 필드는 문제 정의서에서 확인된 바와 같이 "하나의 셀 안에 다수의 검출 건"이 포함될 수 있으므로, 이 분해가 전처리의 최우선 과제다.

**설명가능성(근거) 출력도 이 단계에서 구조적으로 가능해진다.**

- left/masked/right가 분리되어 있으면, 이후 RULE 라벨러가 "어떤 substring을 근거로 삼았는지"를 span(시작/끝 위치)으로 찍어낼 수 있다. 이것이 prediction_evidence 테이블의 기반이 된다.
- 만약 full_context만 있고 구성 요소가 분리되지 않으면, 증거 생성을 위해 동일한 파싱을 추론 시점에 반복해야 하므로 비효율적이고 불일치 위험이 생긴다.

**PK 기반 원본 매핑(문서 핵심 설계)을 구조적으로 보장한다.**

- pk_file과 pk_event를 이 단계에서 생성하여, 이후 어떤 변환(TF-IDF, 정규화, 인코딩)이 적용되더라도 PK를 통해 원본 데이터로 되돌아갈 수 있는 구조를 확립한다.

**"완벽한 파싱"이 아니라 "일관된 파싱"이 핵심이다.**

- 실제 로그 데이터에는 예상치 못한 포맷이 존재한다. 파서가 일부를 놓쳐도, 학습과 추론에서 동일한 파서를 사용하면 모델은 그 일관성을 학습한다. 불일치가 더 위험하다.

**앵커 기반 윈도우링으로 "검출 주변 맥락"에 집중한다.**

- `inspectcontentwithcontext`는 멀티라인/잡음이 섞여 있을 수 있다. 검출 주변 문맥만 쓰면 모델이 "관련 신호"에 집중하고, 새 패턴이 생겨도 문제 정의("검출 주변 맥락")가 유지된다.
- `masked_hit`를 앵커로 `full_context_raw` 내에서 위치를 찾고, 주변 ±W chars를 `local_context_raw`로 추출한다. 못 찾으면 전체의 앞부분을 최대 N chars로 절단한다.
- `file_path`는 짧은 컨텍스트(10자) 문제를 구조적으로 보완하는 핵심 맥락이라 별도로 정규화/파생이 필요하다.

**원문 보존(비파괴)으로 재파생 가능성을 보장한다.**

- `full_context_raw`(원문)와 `full_context_norm`(정규화)를 동시에 보존하여, 새 패턴이 등장했을 때 원문에서 다시 파생할 수 있는 구조를 유지한다.

---
