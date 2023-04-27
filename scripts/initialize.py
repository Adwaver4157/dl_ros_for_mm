#!/usr/bin/python
# -*- coding: utf-8 -*-

import hsrb_interface
import rospy
import sys
from hsrb_interface import geometry

# 移動のタイムアウト[s]
_MOVE_TIMEOUT=60.0
# 把持力[N]
_GRASP_FORCE=0.2
# ボトルのtf名
_BOTTLE_TF='ar_marker/4000'
# グリッパのtf名
_HAND_TF='hand_palm_link'

# ロボット機能を使うための準備
robot = hsrb_interface.Robot()
omni_base = robot.get('omni_base')
whole_body = robot.get('whole_body')
gripper = robot.get('gripper')
tts = robot.get('default_tts')

# bottleのマーカの手前0.02[m],z軸回に-1.57回転させた姿勢
bottle_to_hand = geometry.pose(z=-0.02, ek=-1.57)

# handを0.1[m]上に移動させる姿勢
hand_up = geometry.pose(x=0.1)

# handを0.5[m]手前に移動させる姿勢
hand_back = geometry.pose(z=-0.5)

# ソファの場所
sofa_pos = (1.2, 0.4, 1.57)

if __name__=='__main__':

    # まずは一言
    # rospy.sleep(5.0)
    # tts.say('こんにちはHSRだよ。ソファ脇のペットボトルを掴もうと思います。')
    # rospy.sleep(5.0)

    try:
        # 把持用初期姿勢に遷移
        print("start initialization")
        gripper.command(1.0)
        omni_base.go_abs(0, 0, 0, _MOVE_TIMEOUT)
        whole_body.move_to_neutral()
        print("finished initialization")
    except:
        tts.say('初期化失敗')
        rospy.logerr('fail to init')
        sys.exit()

