from gym.envs.registration import register
from task.Grasp.PandaGraspEnv import PandaGraspEnv
from task.Reach.PandaReachEnv import PandaReachEnv
from task.Touch.PandaTouchEnv import PandaTouchEnv
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

'''
register(
	id='PandaReachEnv_color-v0',
	entry_point='task.Reach.PandaReachEnv:PandaReachEnv',
	kwargs={
		'render': True
	}
)
'''

register(
	id='PandaTouchEnv_color-v0',
	entry_point='task.Touch.PandaTouchEnv:PandaTouchEnv',
	max_episode_steps=250,
	kwargs={
		'render': False
	}
)


'''
register(
	id='PandaTouchEnv_depth-v0',
	entry_point='task.Touch.PandaTouchEnv:PandaTouchEnv',
	kwargs={
		'render': False
	}
)
'''