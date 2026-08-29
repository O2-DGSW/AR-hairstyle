"""server 순수 로직 회귀 테스트.

GPU/모델을 절대 적재하지 않는다: GpuFaceParser / HairFast / FacePose() 인스턴스화
금지. 여기 있는 것은 전부 numpy/cv2 만으로 검증 가능한 기하·상태 로직이다.
"""
