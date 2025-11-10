# main.py
# 하이브리드 추천 파이프라인 실행 스크립트 (모드 지원)

import sys
import os
import argparse

# 프로젝트 루트를 파이썬 경로에 추가하여 모듈을 임포트할 수 있도록 함
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from cbf.run_cbf import run_cbf
from cf.run_cf import run_hybrid_cf, run_cf

def main():
    """
    하이브리드 추천 시스템 파이프라인
    - 'build' 모드: CBF 모델 및 1차 추천 캐시를 생성합니다.
    - 'recommend' 모드: 특정 유저/감정에 대한 최종 추천을 생성합니다.
    """
    parser = argparse.ArgumentParser(description="하이브리드 추천 시스템 파이프라인")
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=["build", "recommend"],
        help="실행 모드를 선택하세요: 'build' 또는 'recommend'"
    )
    parser.add_argument("--user", type=int, help="'recommend' 모드에서 사용할 사용자 ID")
    parser.add_argument("--emotion", type=str, help="'recommend' 모드에서 사용할 감정 라벨")
    parser.add_argument("--top_k", type=int, default=30, help="추천받을 아이템 개수")

    args = parser.parse_args()

    if args.mode == "build":
        print("="*50)
        print("PIPELINE MODE: BUILD")
        print("="*50)
        print("\n>>> STAGE 1: Content-Based Filtering (CBF) 파이프라인 시작...")
        try:
            run_cbf()
            print("\n>>> STAGE 1: Content-Based Filtering (CBF) 성공.")
            print("CBF 1차 추천 캐시가 'output/cbf_intermediate'에 생성되었습니다.")
        except Exception as e:
            print(f"\n>>> STAGE 1: Content-Based Filtering (CBF) 실패: {e}")
            return
        
        # build 시에는 전체 CF까지 실행하여 일괄 추천 결과를 만들어 둘 수도 있습니다.
        # 여기서는 CBF 캐시 생성까지만 build로 정의합니다.
        # print("\n>>> STAGE 2: 전체 사용자에 대한 최종 추천 일괄 생성...")
        # run_hybrid_cf(top_k=args.top_k)
        # print("\n>>> STAGE 2: 일괄 추천 생성 완료.")

    elif args.mode == "recommend":
        if args.user is None or args.emotion is None:
            print("오류: 'recommend' 모드에서는 --user 와 --emotion 인자가 반드시 필요합니다.")
            sys.exit(1)

        print("="*50)
        print(f"PIPELINE MODE: RECOMMEND for User ID: {args.user}, Emotion: {args.emotion}")
        print("="*50)
        
        try:
            # cf/run_cf.py의 단일 유저 추천 함수 호출
            output_path = run_cf(
                user_id=args.user,
                emotion=args.emotion,
                top_k=args.top_k
            )
            print("\n>>> 추천 생성 성공!")
            print(f"결과가 다음 파일에 저장되었습니다: {output_path}")
        except Exception as e:
            print(f"\n>>> 추천 생성 실패: {e}")

if __name__ == "__main__":
    main()