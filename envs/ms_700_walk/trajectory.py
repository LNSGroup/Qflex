import pandas as pd
import scipy
import scipy.signal
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt


class Trajectory:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.load_data()

    def load_data(self):
        df_data = pd.read_csv(self.file_path, sep=",", header=0)
        data_time = df_data["time"]
        mj_joint_data = df_data.drop(columns=["time"]).values
        self.mj_joint_data_tck, self.data_time = self.get_scipy_tck_symmetry(mj_joint_data)
        self.num_joints = mj_joint_data.shape[1]

    def get_scipy_tck_symmetry(self, rawdata):
        time_frame = rawdata.shape[0]
        nq = rawdata.shape[1]

        num_periods = 2  # hardcoded
        mean_period = 114

        # cut the single periods
        startloc = int(np.floor(time_frame / 2 - (num_periods / 2 * mean_period)))
        single_periods = np.zeros((nq, int(mean_period), num_periods))

        for loop1 in range(nq):
            startloc_loop = startloc
            for loop2 in range(num_periods):
                single_periods[loop1, :, loop2] = rawdata[startloc_loop : startloc_loop + mean_period, loop1]
                startloc_loop = startloc_loop + mean_period

        # calculate mean
        mean_gait_angle = np.mean(single_periods, axis=2)
        mean_time = np.array([0.00])
        mean_time = np.append(mean_time, np.linspace(start=0.01, stop=(mean_period - 1) * 0.01, num=mean_period - 1))
        mean_gait_angle_expand = np.concatenate([mean_gait_angle, mean_gait_angle, mean_gait_angle], axis=1)

        b, a = scipy.signal.butter(8, 0.2, 'lowpass')

        for i in range(nq):
            if i == 2:
                # tx
                mean_gait_angle[i, :] = rawdata[startloc : startloc + mean_period, i]
                continue
            mean_gait_angle_expand[i, :] = scipy.signal.filtfilt(b, a, mean_gait_angle_expand[i, :])
            mean_gait_angle[i, :] = mean_gait_angle_expand[i, mean_period : 2 * mean_period]

        temp_p = np.expand_dims(mean_gait_angle[:, 0], axis=0)

        mean_time = np.array([0.00])
        mean_time = np.append(mean_time, np.linspace(start=0.01, stop=(mean_period) * 0.01, num=mean_period))
        mean_gait_angle = np.concatenate([mean_gait_angle, temp_p.T], axis=1)

        for i in range(3):
            mean_gait_angle[i, :] = mean_gait_angle[i, :] - mean_gait_angle[i, 0]

        tck_list = []
        for i in range(nq):
            if i == 2:  # tx
                mean_gait_angle[i, -1] = mean_gait_angle[i, -2] + mean_gait_angle[i, 1] - mean_gait_angle[i, 0]
                tck = interpolate.splrep(mean_time, mean_gait_angle[i, :], s=0)
            else:
                tck = interpolate.splrep(mean_time, mean_gait_angle[i, :], s=0, per=1)
            tck_list.append(tck)

        data_time_length = mean_period * 0.01

        return tck_list, data_time_length

    def query(self, time):
        period_now = time // self.data_time
        t_interp = time % self.data_time
        p_interp = np.zeros(self.num_joints)
        v_interp = np.zeros(self.num_joints)

        for i in range(self.num_joints):
            tck_now = self.mj_joint_data_tck[i]
            p_interp[i] = interpolate.splev(t_interp, tck_now, der=0)
            v_interp[i] = interpolate.splev(t_interp, tck_now, der=1)

        # tx
        p_interp[2] += interpolate.splev(self.data_time, self.mj_joint_data_tck[2], der=0) * period_now

        return p_interp, v_interp



class Trajectory_IK:
    def __init__(self, qpos_file_path, emg_file_path=None):
        self.qpos_file_path = qpos_file_path
        self.emg_file_path = emg_file_path
        self.load_qpos()
        if self.emg_file_path is not None:
            self.load_emg()
        
    def load_qpos(self):
        df_data = pd.read_csv(self.qpos_file_path, sep="\t", header=6)
        data_time_list = df_data["time"].values

        mj_joint_data = np.stack(
            (
                # pelvis joints
                np.array(df_data["pelvis_tz"]),         # 0
                np.array(df_data["pelvis_ty"]) - 0.97,  # mujoco model pelvis is init to 0.97m height
                np.array(df_data["pelvis_tx"]),
                np.array(df_data["pelvis_tilt"]),
                np.array(df_data["pelvis_list"]),
                np.array(df_data["pelvis_rotation"]),
                # lowerbody joints
                np.array(df_data["hip_flexion_r"]),     # 6
                np.array(df_data["hip_adduction_r"]),
                np.array(df_data["hip_rotation_r"]),
                np.array(df_data["knee_angle_r"]),
                np.array(df_data["ankle_flexion_r"]),
                np.array(df_data["ankle_in_ev_r"]),
                np.array(df_data["ankle_rot_r"]),
                np.array(df_data["subtalar_angle_r"]),
                np.array(df_data["hip_flexion_l"]),
                np.array(df_data["hip_adduction_l"]),
                np.array(df_data["hip_rotation_l"]),
                np.array(df_data["knee_angle_l"]),
                np.array(df_data["ankle_flexion_l"]),
                np.array(df_data["ankle_in_ev_l"]),
                np.array(df_data["ankle_rot_l"]),
                np.array(df_data["subtalar_angle_l"]),
                # torso joints
                np.array(df_data["lumbar_FE"]),         # 22
                np.array(df_data["lumbar_LB"]),
                np.array(df_data["lumbar_AR"]),
                np.array(df_data["thoracic_FE"]),
                np.array(df_data["thoracic_LB"]),
                np.array(df_data["thoracic_AR"]),
                np.array(df_data["cervical_FE"]),
                np.array(df_data["cervical_LB"]),
                np.array(df_data["cervical_AR"]),
                np.array(df_data["head_FE"]),
                np.array(df_data["head_LB"]),
                np.array(df_data["head_AR"]),
                # arm joints
                np.array(df_data["elv_angle_r"]),       # 34
                np.array(df_data["shoulder_elv_r"]),
                np.array(df_data["shoulder_rot_r"]),
                np.array(df_data["elbow_flexion_r"]),
                np.array(df_data["pro_sup_r"]),
                np.array(df_data["elv_angle_l"]),
                np.array(df_data["shoulder_elv_l"]),
                np.array(df_data["shoulder_rot_l"]),
                np.array(df_data["elbow_flexion_l"]),
                np.array(df_data["pro_sup_l"]),
            ),
            axis=1,
        )

        # start from 0
        mj_joint_data[:, 0] = mj_joint_data[:, 0] - mj_joint_data[0, 0]
        mj_joint_data[:, 2] = mj_joint_data[:, 2] - mj_joint_data[0, 2]

        # degree to rad
        mj_joint_data[:, 3:] = mj_joint_data[:, 3:] / 180 * np.pi

        self.num_joints = mj_joint_data.shape[1]
        self.mj_joint_data_tck, self.data_time = self.get_scipy_tck(data_time_list, mj_joint_data)

        self.mj_joint_data_tck_expand, self.data_time_expand = self.get_scipy_tck_expand(data_time_list, mj_joint_data)

        # print('trc time: ', data_time_list[-1])

    def load_emg(self):
        df_data = pd.read_csv(self.emg_file_path, sep="\t", header=0)
        self.emg_time_list = df_data["Time"].values
        self.emg_name_list = df_data.columns[2:]
        self.emg_data = df_data[self.emg_name_list].values

        # expand to 3 periods
        periods = 3
        emg_time_length = self.emg_time_list.shape[0] * periods * (self.emg_time_list[1] - self.emg_time_list[0])
        self.emg_time_list_expand = np.arange(0, emg_time_length, self.emg_time_list[1] - self.emg_time_list[0])
        self.emg_data_expand = np.concatenate([self.emg_data] * periods, axis=0)

        self.emg_fs = int(1 / (self.emg_time_list[1] - self.emg_time_list[0]))  # 800 Hz

        self.emg_envelope_data = self.emg_envelope(self.emg_data, self.emg_fs)

        self.emg_envelope_data_expand = self.emg_envelope(self.emg_data_expand, self.emg_fs)

        print('EMG time: ', self.emg_time_list[-1])
        print('EMG fs: ', self.emg_fs)

    def get_scipy_tck_expand(self, data_time_list, rawdata):
        data_time_length = data_time_list[-1]
        # expand to 3 periods
        data_time_length_expand = data_time_list.shape[0] * 3 * (data_time_list[1] - data_time_list[0])
        data_time_list_expand = np.arange(0, data_time_length_expand, data_time_list[1] - data_time_list[0])
        data_expand = np.concatenate((rawdata, rawdata, rawdata), axis=0)
        
        b, a = scipy.signal.butter(8, 0.2, 'lowpass')
        for i in range(rawdata.shape[1]):
            if i == 2:
                # tx
                continue
            data_expand[:, i] = scipy.signal.filtfilt(b, a, data_expand[:, i])

        # extract the middle period
        period_data = data_expand[data_time_list.shape[0] : data_time_list.shape[0] * 2, :]

        tck_list = []
        for i in range(rawdata.shape[1]):
            if i == 2:
                # tx
                tck = interpolate.splrep(data_time_list, period_data[:, i], s=0)
            else:
                tck = interpolate.splrep(data_time_list, period_data[:, i], s=0, per=1)
            tck_list.append(tck)

        return tck_list, data_time_length
    
    def get_scipy_tck(self, data_time_list, rawdata):
        b, a = scipy.signal.butter(8, 0.2, 'lowpass')
        for i in range(rawdata.shape[1]):
            rawdata[:, i] = scipy.signal.filtfilt(b, a, rawdata[:, i])

        tck_list = []
        for i in range(rawdata.shape[1]):
            tck = interpolate.splrep(data_time_list, rawdata[:, i], s=0)
            tck_list.append(tck)

        data_time_length = data_time_list[-1]

        return tck_list, data_time_length

    def query(self, time, expanded=False):
        
        if time < 0:
            raise ValueError("time out of trajectory range")
        
        p_interp = np.zeros(self.num_joints)
        v_interp = np.zeros(self.num_joints)
        
        if expanded:
            period_now = time // self.data_time
            t_interp = time % self.data_time
            for i in range(self.num_joints):
                tck_now = self.mj_joint_data_tck_expand[i]
                p_interp[i] = interpolate.splev(t_interp, tck_now, der=0)
                v_interp[i] = interpolate.splev(t_interp, tck_now, der=1)

            # tx
            p_interp[2] += interpolate.splev(self.data_time, self.mj_joint_data_tck_expand[2], der=0) * period_now

        else:
            for i in range(self.num_joints):
                tck_now = self.mj_joint_data_tck[i]
                p_interp[i] = interpolate.splev(time, tck_now, der=0)
                v_interp[i] = interpolate.splev(time, tck_now, der=1)

        return p_interp, v_interp
    
    def emg_envelope(self, emg_data, fs):
        # mean
        emg_data = emg_data - np.mean(emg_data, axis=0)
        
        # 4 order high-pass filter
        # 50Hz
        Wn_high = 50 / (fs / 2)
        b, a = scipy.signal.butter(4, Wn_high, 'highpass')
        emg_data_band_pass = scipy.signal.filtfilt(b, a, emg_data, axis=0)

        # rectified
        emg_data_rectified = np.abs(emg_data_band_pass)

        # 4 order low-pass filter
        # 8Hz
        Wn_low = 8 / (fs / 2)
        b, a = scipy.signal.butter(4, Wn_low, 'lowpass')
        emg_data_low_pass = scipy.signal.filtfilt(b, a, emg_data_rectified, axis=0)

        # gaussian somoothed with a 100 ms window
        # 100ms
        window_length = int(0.1 * fs)
        emg_data_envelope = np.zeros_like(emg_data_low_pass)
        for i in range(emg_data_low_pass.shape[1]):
            emg_data_envelope[:, i] = scipy.ndimage.gaussian_filter1d(emg_data_low_pass[:, i], window_length)

        # using max to normalize
        emg_data_envelope = emg_data_envelope / np.max(emg_data_envelope, axis=0)

        return emg_data_envelope

    def plot(self):
        # plot the p interpolation
        plt_dict_pelvis = {'pelvis_tz': 0, 'pelvis_ty': 1, 'pelvis_tx': 2, 
                    'pelvis_tilt': 3, 'pelvis_list': 4, 'pelvis_rotation': 5,}
        plt_dict = {'hip_flexion_r': 6, 'hip_adduction_r': 7, 'knee_angle_r': 9, 'ankle_flexion_r': 10, 
                    'hip_flexion_l': 14, 'hip_adduction_l': 15, 'knee_angle_l': 17, 'ankle_flexion_l': 18, 
                    'elv_angle_r': 34, 'shoulder_elv_r': 35, 'shoulder_rot_r': 36, 'elbow_flexion_r': 37, 
                    'elv_angle_l': 39, 'shoulder_elv_l': 40, 'shoulder_rot_l': 41, 'elbow_flexion_l': 42}
        
        # 0.01s
        t = np.arange(0, self.data_time, 0.01)
        t_expand = np.arange(0, self.data_time * 3, 0.01)

        plt.figure(figsize=(10, 8))
        i = 0
        for label, index in plt_dict_pelvis.items():
            plt.subplot(2, 3, i+1)
            data = np.zeros_like(t)
            data_expand = np.zeros_like(t_expand)
            for j in range(len(t)):
                data[j] = self.query(t[j], expanded=False)[0][index]
            for j in range(len(t_expand)):
                data_expand[j] = self.query(t_expand[j], expanded=True)[0][index]

            plt.plot(t_expand, data_expand, label='expanded', linewidth=1)
            plt.plot(t, data, label='original', linewidth=2)
            plt.title(label)
            plt.legend()
            i += 1
        
        plt.figure(figsize=(20, 10))
        i = 0
        for label, index in plt_dict.items():
            plt.subplot(4, 4, i+1)
            data = np.zeros_like(t)
            data_expand = np.zeros_like(t_expand)
            for j in range(len(t)):
                data[j] = self.query(t[j], expanded=False)[0][index]
            for j in range(len(t_expand)):
                data_expand[j] = self.query(t_expand[j], expanded=True)[0][index]

            plt.plot(t_expand, data_expand, label='expanded', linewidth=1)
            plt.plot(t, data, label='original', linewidth=2)
            plt.title(label)
            plt.legend()
            i += 1
            
        plt.show()

    def plot_emg(self):
        # plot the p interpolation
        qpos_plt_dict = {'hip_flexion_r': 6, 'hip_adduction_r': 7, 'knee_angle_r': 9, 'ankle_flexion_r': 10}
        
        # 0.01s
        t = np.arange(0, self.data_time, 0.01)
        t_expand = np.arange(0, self.data_time * 3, 0.01)
        
        plt.figure(figsize=(20, 10))
        i = 0
        for label, index in qpos_plt_dict.items():
            plt.subplot(5, 4, i+1)
            data = np.zeros_like(t)
            data_expand = np.zeros_like(t_expand)
            for j in range(len(t)):
                data[j] = self.query(t[j], expanded=False)[0][index]
            for j in range(len(t_expand)):
                data_expand[j] = self.query(t_expand[j], expanded=True)[0][index]

            plt.plot(t_expand, data_expand, label='expanded', linewidth=1)
            plt.plot(t, data, label='original', linewidth=2)
            plt.title(label)
            # axvline = 0.09
            plt.axvline(x=0.09, color='r', linestyle='--')
            plt.legend()
            i += 1

        for i in range(len(self.emg_name_list)):
            index = i + 5 + i // 4 * 4
            plt.subplot(5, 4, index)
            plt.plot(self.emg_time_list_expand, self.emg_data_expand[:, i], label='expanded', linewidth=1)
            plt.plot(self.emg_time_list, self.emg_data[:, i], label='original', linewidth=2)
            plt.title(self.emg_name_list[i])
            plt.axvline(x=0.09, color='r', linestyle='--')
            plt.legend()

            plt.subplot(5, 4, index+4)
            # plt.subplot(3, 4, i+5)
            plt.plot(self.emg_time_list_expand, self.emg_envelope_data_expand[:, i], label='expanded', linewidth=1)
            plt.plot(self.emg_time_list, self.emg_envelope_data[:, i], label='original', linewidth=2)
            plt.ylim([0, 1.2])
            plt.title(self.emg_name_list[i] + ' envelope')
            plt.axvline(x=0.09, color='r', linestyle='--')
            plt.legend()

        # # plot the envelope
        # for i in range(len(self.emg_name_list)):
        #     plt.subplot(5, 4, i+13)
        #     # plt.subplot(3, 4, i+5)
        #     plt.plot(self.emg_time_list_expand, self.emg_envelope_data_expand[:, i], label='expanded', linewidth=1)
        #     plt.plot(self.emg_time_list, self.emg_envelope_data[:, i], label='original', linewidth=2)
        #     plt.ylim([0, 1.2])
        #     plt.title(self.emg_name_list[i] + ' envelope')
        #     plt.axvline(x=0.09, color='r', linestyle='--')
        #     plt.legend()
            
        plt.show()

class Trajectory_IK_700(Trajectory_IK):
    def load_qpos(self):
        df_data = pd.read_csv(self.qpos_file_path, sep="\t", header=6)
        data_time_list = df_data["time"].values

        mj_joint_data = np.stack(
            (
                # pelvis joints
                np.array(df_data["pelvis_tz"]),         # 0
                np.array(df_data["pelvis_ty"]) - 0.97,  # mujoco model pelvis is init to 0.97m height
                np.array(df_data["pelvis_tx"]),
                np.array(df_data["pelvis_tilt"]),
                np.array(df_data["pelvis_list"]),
                np.array(df_data["pelvis_rotation"]),
                # lowerbody joints
                np.array(df_data["hip_flexion_r"]),     # 6
                np.array(df_data["hip_adduction_r"]),
                np.array(df_data["hip_rotation_r"]),
                np.array(df_data["knee_angle_r"]),
                np.array(-1*df_data["ankle_flexion_r"]),
                # np.array(df_data["ankle_in_ev_r"]),
                # np.array(df_data["ankle_rot_r"]),
                np.array(df_data["subtalar_angle_r"]),
                np.array(df_data["hip_flexion_l"]),
                np.array(df_data["hip_adduction_l"]),
                np.array(df_data["hip_rotation_l"]),
                np.array(df_data["knee_angle_l"]),
                np.array(-1*df_data["ankle_flexion_l"]),
                # np.array(df_data["ankle_in_ev_l"]),
                # np.array(df_data["ankle_rot_l"]),
                np.array(df_data["subtalar_angle_l"]),
                # torso joints
                np.array(df_data["lumbar_FE"]),         # 22
                np.array(df_data["lumbar_LB"]),
                np.array(df_data["lumbar_AR"]),
                np.array(df_data["thoracic_FE"]),
                np.array(df_data["thoracic_LB"]),
                np.array(df_data["thoracic_AR"]),
                np.array(df_data["cervical_FE"]),
                np.array(df_data["cervical_LB"]),
                np.array(df_data["cervical_AR"]),
                # np.array(df_data["head_FE"]),
                # np.array(df_data["head_LB"]),
                # np.array(df_data["head_AR"]),
                # arm joints
                np.array(df_data["elv_angle_r"]),       # 34
                np.array(df_data["shoulder_elv_r"]),
                np.array(df_data["shoulder_rot_r"]),
                np.array(df_data["elbow_flexion_r"]),
                np.array(df_data["pro_sup_r"]),
                np.array(df_data["elv_angle_l"]),
                np.array(df_data["shoulder_elv_l"]),
                np.array(df_data["shoulder_rot_l"]),
                np.array(df_data["elbow_flexion_l"]),
                np.array(df_data["pro_sup_l"]),
            ),
            axis=1,
        )

        # start from 0
        mj_joint_data[:, 0] = mj_joint_data[:, 0] - mj_joint_data[0, 0]
        mj_joint_data[:, 2] = mj_joint_data[:, 2] - mj_joint_data[0, 2]

        # degree to rad
        mj_joint_data[:, 3:] = mj_joint_data[:, 3:] / 180 * np.pi

        self.num_joints = mj_joint_data.shape[1]
        self.mj_joint_data_tck, self.data_time = self.get_scipy_tck(data_time_list, mj_joint_data)

        self.mj_joint_data_tck_expand, self.data_time_expand = self.get_scipy_tck_expand(data_time_list, mj_joint_data)
# main function
# if __name__ == "__main__":
#     traj = Trajectory_IK(qpos_file_path="ms_envs\msmodel_gym\envs\motion_data\Subject46_Walking_6.mot", emg_file_path="ms_envs\msmodel_gym\envs\motion_data\Subject46_Walking_6_EMG.csv")
#     # traj.plot()
#     traj.plot_emg()