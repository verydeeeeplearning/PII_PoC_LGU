# 07. PoC 산출물 템플릿

> **문서 버전**: 1.2
> **최종 수정일**: 2026-02-22
> **기반 문서**: docs/Architecture/ v1.2 §10, §11, §12, §17, §18, §22

## 1. 목적

본 문서는 Server-i 오탐 개선 **PoC 범위**에서 필요한 산출물만 정의합니다.

## 2. 필수 산출물 (v1.2)

### 2.1 데이터 레이어 산출물

| 레이어 | 파일 | 설명 |
|--------|------|------|
| **Bronze (S0)** | `data/processed/bronze_events.parquet` | 원본 Dataset A, 컬럼명 정규화만 적용 |
| **Bronze (S0)** | `data/processed/bronze_labels.parquet` | 원본 Dataset B, 컬럼명 정규화만 적용 |
| **Silver-S1** | `data/processed/silver_detections.parquet` | 1행=1검출 이벤트, pk_file/pk_event 포함 |
| **Silver-S1** | `data/processed/silver_quarantine.parquet` | 파싱 실패/스키마 불일치 격리 행 |
| **Silver-S2** | Feature 컬럼 추가된 silver_detections | raw_text, shape_text, path_text, tabular 피처 포함 |

### 2.2 모델 및 평가 산출물

| 구분 | 파일/형식 | 필수 내용 |
|------|-----------|----------|
| 성능 보고서 | `outputs/reports/classification_report.txt` | 클래스별 Precision/Recall/F1, Macro F1 (pk_file Group + Time Split 기준) |
| 혼동행렬 | `outputs/figures/confusion_matrix.png` | 8개 클래스 기준 오분류 분포 |
| Feature 중요도 | `outputs/reports/feature_importance.csv` | 상위 중요 Feature 목록 |
| 모델 파일 | `models/final/best_model_v1.joblib` | 최종 선택 ML Labeler 모델 1종 |
| 실행 설정 | `config/feature_config.yaml`, `config/model_config.yaml` | 재현 가능한 학습/평가 설정 |

### 2.3 예측 출력 산출물 (Stage S5)

v1.2 출력은 `predictions_main` + `prediction_evidence` 두 테이블로 구성된다.

**predictions_main.parquet** (1행 = 1검출 이벤트)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `pk_event` | string | 검출 이벤트 고유 ID (원본 조인용) |
| `pk_file` | string | 파일 단위 고유 ID |
| `final_label` | string | 최종 예측 라벨 (8클래스 또는 UNKNOWN) |
| `decision_source` | string | 라벨 출처: `RULE` / `ML` / `HYBRID` |
| `confidence` | float | 예측 신뢰도 (0~1) |
| `risk_flag` | int | TP 오분류 위험 플래그 (0/1) |
| `ood_flag` | int | Out-Of-Distribution 플래그 (0/1) |
| `reason_code` | string | 사유 코드 (예: `OS_COPYRIGHT_DOMAIN`, `EPOCH_TIMESTAMP`) |

**prediction_evidence.parquet** (N행 = 1검출 이벤트, 근거 상세)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `pk_event` | string | predictions_main 조인 키 |
| `evidence_type` | string | 근거 유형: `RULE_MATCH` / `ML_PROBA` / `FEATURE_VALUE` |
| `evidence_key` | string | 근거 내용 (예: 매칭된 도메인, 피처명) |
| `evidence_value` | string | 근거 값 (예: `@redhat.com`, `0.87`) |
| `evidence_weight` | float | 해당 근거의 신뢰도 기여도 |

## 3. PoC 결과 보고 템플릿 (v1.2)

### 3.1 정량 결과

- 평가 분할 방식: `pk_file Group Split` / `Time Split` (해당 적용 방식 표시)
- Macro F1: `__`
- TP Recall: `__`
- FP Precision: `__`
- RULE Labeler 커버리지: `__` %
- UNKNOWN 비율: `__` %
- parse_success_rate: `__` %
- PoC 기준 충족 여부: `PASS/FAIL`

### 3.2 정성 결과

- 주요 오탐 유형 분류 가능 여부: `__`
- 오분류 상위 패턴: `__`
- UNKNOWN 케이스 특징 (OOD 원인 추정): `__`
- 즉시 개선 필요 항목: `__`

## 4. 제출 전 체크리스트

- [ ] 최종 모델 1종 저장 완료
- [ ] bronze/silver Parquet 파일 생성 완료
- [ ] predictions_main + prediction_evidence 생성 완료
- [ ] 평가 리포트 및 시각화 생성 완료 (pk_file Group + Time Split 기준)
- [ ] 설정 파일(`config/*.yaml`) 최신화 완료
- [ ] 자동화 설정 파일(`label_governance.yaml`, 임계값 설정) 최신화 완료
- [ ] Auto-Mapper/Auto-Taxonomy-Manager/Auto-Retrainer 로그 저장 완료
- [ ] Confident Learning 라벨 품질 감사 리포트 저장 완료
- [ ] 재실행 명령어 검증 완료
- [ ] 전송 경로 문구 확인 완료

## 5. 운영 자동화 산출물 (v1.2 확장)

### 5.1 라벨 거버넌스 산출물

| 구분 | 파일 | 필수 내용 |
|------|------|----------|
| 거버넌스 규칙 | `config/label_governance.yaml` | Canonical 8+UNKNOWN 매핑 규칙, 유사도 임계값, 자동 조치 정책 |
| 라벨 매핑 테이블 | `config/label_mapping.yaml` | 신규 라벨 -> Primary Class/Reason Code 매핑 이력 |
| Auto-Mapper 리포트 | `outputs/reports/auto_mapper_report.json` | 신규 라벨 매핑 결과, similarity score, UNKNOWN 전달 건수 |
| Auto-Taxonomy-Manager 리포트 | `outputs/reports/auto_taxonomy_report.json` | UNKNOWN 클러스터링 결과, Reason Code 확장/신규 클래스 제안 내역 |

### 5.2 자동 판정/재학습 산출물

| 구분 | 파일 | 필수 내용 |
|------|------|----------|
| Auto-Adjudicator 로그 | `outputs/reports/auto_adjudication_log.parquet` | NEEDS_REVIEW 자동 판정 단계(1~4), 최종 reason_code |
| UNKNOWN 자동 처리 로그 | `outputs/reports/unknown_auto_process_log.parquet` | `OOD_AUTO_ASSIGN` / `OOD_UNKNOWN_TO_TP` 처리 결과 |
| Auto-Tuner 로그 | `outputs/reports/auto_tuner_history.json` | TAU 후보 탐색 결과, 제약 충족 여부, 최종 채택값 |
| Auto-Retrainer 이력 | `outputs/reports/auto_retrainer_history.json` | 트리거 원인, 재학습 결과, 롤백/채택 판단 |

### 5.3 라벨 품질 감사 산출물 (Confident Learning)

| 구분 | 파일 | 필수 내용 |
|------|------|----------|
| 감사 결과 | `outputs/reports/confident_learning_audit.json` | `noise_rate`, `error_count`, 정제 필요 여부 |
| 정제 대상 목록 | `outputs/reports/label_error_candidates.parquet` | 고확신 불일치 샘플 인덱스/키 |
| 정제 데이터 버전 | `data/processed/silver_labels_cleaned.parquet` | 정제 적용 후 학습 라벨셋 |

전송 경로:
`Google Drive -> LGU PC 로컬환경 다운로드 -> 자료전송 시스템(업무 클라우드) -> 자료전송 시스템(보안 클라우드) -> SSH를 통해 파일 전송`
