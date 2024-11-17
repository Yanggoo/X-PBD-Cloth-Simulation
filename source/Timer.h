#pragma once
#include <GL/glut.h>
namespace ClothSimulation {
	class Timer {
	public:
		Timer(float FixedUpdateInterval = 0.016f) // 默认固定更新间隔为 0.016 秒（即 60 次每秒）
			: m_FixedUpdateInterval(FixedUpdateInterval),
			m_PreviousTime(0.0f),
			m_DeltaTime(0.0f),
			m_FixedTimeAccumulator(0.0f)
		{
			// 初始化起始时间
			m_PreviousTime = GetCurrentTime();
		}

		// 更新 DeltaTime 和累积的 FixedUpdate 时间
		void Update() {
			float CurrentTime = GetCurrentTime();
			m_DeltaTime = CurrentTime - m_PreviousTime; // 计算 DeltaTime
			m_PreviousTime = CurrentTime;

			m_FixedTimeAccumulator += m_DeltaTime; // 累积时间，用于 FixedUpdate
		}

		// 检查是否需要进行固定时间步更新
		bool NeedsFixedUpdate() {
			bool need = false;
			while (m_FixedTimeAccumulator >= m_FixedUpdateInterval) {
				m_FixedTimeAccumulator -= m_FixedUpdateInterval; // 减去固定更新间隔
				need = true;
			}
			return need;
		}

		// 获取 DeltaTime
		float GetDeltaTime() const { return m_DeltaTime; }

		// 获取固定更新间隔
		float GetFixedUpdateInterval() const { return m_FixedUpdateInterval; }

		void Reset() {
			m_PreviousTime = GetCurrentTime();
			m_DeltaTime = 0.0f;
			m_FixedTimeAccumulator = 0.0f;
		}

	private:
		float GetCurrentTime() const {
			return static_cast<float>(glutGet(GLUT_ELAPSED_TIME)) / 1000.0f; // 获取当前时间，单位为秒
		}

		float m_FixedUpdateInterval;  // 固定更新间隔（秒）
		float m_PreviousTime;         // 上一帧的时间
		float m_DeltaTime;            // 帧间隔时间
		float m_FixedTimeAccumulator; // 累积的固定更新时间
	};

}
