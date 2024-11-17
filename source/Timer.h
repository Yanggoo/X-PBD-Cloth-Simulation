#pragma once
#include <GL/glut.h>
namespace ClothSimulation {
	class Timer {
	public:
		Timer(float FixedUpdateInterval = 0.016f) // Ĭ�Ϲ̶����¼��Ϊ 0.016 �루�� 60 ��ÿ�룩
			: m_FixedUpdateInterval(FixedUpdateInterval),
			m_PreviousTime(0.0f),
			m_DeltaTime(0.0f),
			m_FixedTimeAccumulator(0.0f)
		{
			// ��ʼ����ʼʱ��
			m_PreviousTime = GetCurrentTime();
		}

		// ���� DeltaTime ���ۻ��� FixedUpdate ʱ��
		void Update() {
			float CurrentTime = GetCurrentTime();
			m_DeltaTime = CurrentTime - m_PreviousTime; // ���� DeltaTime
			m_PreviousTime = CurrentTime;

			m_FixedTimeAccumulator += m_DeltaTime; // �ۻ�ʱ�䣬���� FixedUpdate
		}

		// ����Ƿ���Ҫ���й̶�ʱ�䲽����
		bool NeedsFixedUpdate() {
			bool need = false;
			while (m_FixedTimeAccumulator >= m_FixedUpdateInterval) {
				m_FixedTimeAccumulator -= m_FixedUpdateInterval; // ��ȥ�̶����¼��
				need = true;
			}
			return need;
		}

		// ��ȡ DeltaTime
		float GetDeltaTime() const { return m_DeltaTime; }

		// ��ȡ�̶����¼��
		float GetFixedUpdateInterval() const { return m_FixedUpdateInterval; }

		void Reset() {
			m_PreviousTime = GetCurrentTime();
			m_DeltaTime = 0.0f;
			m_FixedTimeAccumulator = 0.0f;
		}

	private:
		float GetCurrentTime() const {
			return static_cast<float>(glutGet(GLUT_ELAPSED_TIME)) / 1000.0f; // ��ȡ��ǰʱ�䣬��λΪ��
		}

		float m_FixedUpdateInterval;  // �̶����¼�����룩
		float m_PreviousTime;         // ��һ֡��ʱ��
		float m_DeltaTime;            // ֡���ʱ��
		float m_FixedTimeAccumulator; // �ۻ��Ĺ̶�����ʱ��
	};

}
