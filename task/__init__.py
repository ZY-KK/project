from gym.envs.registration import register
from task.Grasp.PandaGraspEnv import PandaGraspEnv
from task.Reach.PandaReachEnv import PandaReachEnv
from bullet.pyBullet import PyBullet
'''
register(
	id='PandaGraspEnv-v0',
	entry_point='task.Grasp.PandaGraspEnv:PandaGraspEnv',
	kwargs={
		'render': True
	}
)
'''
'''
register(
	id='PandaGraspEnv_color-v0',
	entry_point='task.Grasp.PandaGraspEnv:PandaGraspEnv',
	kwargs={
		'render': True
	}
)
'''


register(
	id='PandaReachEnv_color-v0',
	entry_point='task.Reach.PandaReachEnv:PandaReachEnv',
	kwargs={
		'render': True
	}
)
