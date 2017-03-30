import model_pb2
import google.protobuf.text_format


#for model in param.models:
#    print(model)

latency_limit = 1000 # 2secs
RTT = 100 # 100ms
server_cost = 1.8e-7 # $ per ms
send_energy = 3.8/1000 # 3.8mJ

def check_server(x, RTT, latency_limit):
    return x.s_loading_latency + x.s_compute_latency + RTT <= latency_limit

def check_server_cost(x, cost_budget, remaining_time, server_cost, freq, sumsqfreq):
    return freq * (cost_budget / remaining_time / (sumsqfreq + freq*freq)) >= (x.s_compute_latency * server_cost)

def check_device_cost(x, energy_budget, remaining_time, is_away, time_from_last_swap, freq, sumsqfreq):
    e_b = energy_budget
    if is_away:
        e_b -= x.loading_energy * (remaining_time / float(time_from_last_swap))
    return freq * (e_b / remaining_time / (sumsqfreq + freq*freq)) >= (x.compute_energy)

def check_split_cost(x, energy_budget, cost_budget, remaining_time, is_away, time_from_last_swap, freq):
    e_b = energy_budget
    if is_away:
        e_b -= x.sp_loading_energy * (remaining_time / float(time_from_last_swap))
    r1 = (e_b / remaining_time / freq) >= (x.sp_compute_energy)
    r2 = (cost_budget / remaining_time / freq) >= (x.sp_s_compute_latency * server_cost)
    return r1 and r2

def check_device(x, latency_limit):
    return x.loading_latency + x.compute_latency + RTT < latency_limit

def check_split(x, RTT, latency_limit):
    return max(x.sp_loading_latency, x.sp_s_loading_latency) + x.sp_compute_latency + x.sp_s_compute_latency + RTT < latency_limit


class Location:
    NOTRUNNING, DEVICE, SPLIT, SERVER = range(0,4)

class AppType:
    FACE, OBJECT, SCENE, FACE_V = range(4)

class Application:
    def __init__(self, name, freq, models, special_models=[]):
        self.name = name
        self.freq = float(freq)
        self.models = models
        self.status = Location.NOTRUNNING
        self.last_swapin = 0
        self.last_swapin_split = 0
        self.res = []
        self.specializable = len(special_models)>0 
        self.special_models = special_models
        self.pick = None


import bisect 
stat_duration = 5*60 # 5min

import collections

class Scheduler:
    def __init__(self, name, energy_budget, cost_budget):
        self.name = name
        self.energy_budget = float(energy_budget)
        self.cost_budget = float(cost_budget)
        self.applications = collections.defaultdict(list) 
        self.res = []
        self.connectivity = [(0,True)]
        self.connectivity_times = [0]
        self.use_split = False
        self.use_specialize = True
        self.count_stat = [] 
        self.in_context = 0
        self.out_context = 0
        self.in_cache = {}
        self.in_cache[Location.DEVICE] = []
        self.in_cache[Location.SERVER] = []
        self.use_sharing = False
        self.server_only = False
        self.client_only = False

    def add_application(self, app_type, application):
        self.applications[app_type].append(application)

    def set_connectivity(self, conn):
        self.connectivity = conn
        self.connectivity_times = map(lambda x:x[0], conn)

    def get_connectivity(self, i):
        j = bisect.bisect_left(self.connectivity_times, i)
        return self.connectivity[j-1][1]

    def execute_task(self, i, until, tApp, cur, use_sharing=False):
        if use_sharing:
            sharing_app = self.applications[AppType.FACE][0]
            if sharing_app.pick != None:
                name = sharing_app.pick.name
                for model in tApp.models:
                    if name == model.name:
                        tApp.res.append((i, model.accuracy, sharing_app.status))
                        return

        special_models = []
        if self.use_specialize and i>stat_duration:
            self.count_stat.append((i, cur[1]))
            if cur[1] < 7:
                self.in_context += 1
            else:
                self.out_context += 1

            while self.count_stat[0][0] < i-stat_duration:
                p = self.count_stat.pop(0)
                if p[1] < 7:
                    self.in_context -= 1
                else:
                    self.out_context -= 1
            cur_per = self.in_context / float(self.in_context + self.out_context) 
            if cur_per >= 0.6:
                special_models = tApp.special_models
                for model in special_models:
                    model.accuracy = 0
                    for sp in model.special:
                        if cur_per < sp.percent: break
                        if sp.accuracy > model.accuracy:
                            model.accuracy = sp.accuracy

        connected = False
        if (not self.client_only) and self.get_connectivity(i):
            connected = True
            if len(special_models) > 0:
                target_s = special_models
            else:
                target_s = tApp.models
        else:
            target_s = []

        if self.server_only:
            target_c = []
        else:
            if len(special_models) > 0:
                target_c = special_models
            else:
                target_c = tApp.models
        if self.use_split:
            target_sp = tApp.models + special_models
        # server_side
        if tApp.status == Location.NOTRUNNING: # cold miss
            target_s = filter(lambda x: check_server(x, RTT, latency_limit), target_s) 
            target_c = filter(lambda x: check_device(x, latency_limit), target_c) 
            if self.use_split:
                target_sp = filter(lambda x: check_split(x, RTT, latency_limit), target_sp)

        freqsqsum = sum(map(lambda x:x.freq*x.freq, self.in_cache[Location.SERVER]))
        if tApp.status == Location.SERVER:
            freqsqsum -= (tApp.freq * tApp.freq)

        target_s = filter(lambda x: check_server_cost(x, self.cost_budget, until-i, server_cost, tApp.freq, freqsqsum), target_s)

        freqsqsum = sum(map(lambda x:x.freq*x.freq, self.in_cache[Location.DEVICE]))
        if tApp.status == Location.DEVICE:
            freqsqsum -= (tApp.freq * tApp.freq)

        target_c = filter(lambda x: check_device_cost(x, self.energy_budget, until-i, tApp.status==Location.SERVER, 
            (i-tApp.last_swapin), tApp.freq, freqsqsum), target_c)
        if self.use_split:
            target_sp = filter(lambda x: check_split_cost(x, self.energy_budget, self.cost_budget, until-i, 
                tApp.status==Location.SERVER or tApp.status==Location.DEVICE, (i-tApp.last_swapin_split), tApp.freq), target_sp)

        target_s.sort(key=lambda x:x.accuracy, reverse=True)
        target_c.sort(key=lambda x:x.accuracy, reverse=True)
        if self.use_split:
            target_sp.sort(key=lambda x:x.accuracy, reverse=True)

        picks = []
        try: 
            server_pick = target_s[0]
            server_pick.location = Location.SERVER
            picks.append(server_pick)
        except:
            server_pick = None
        try:
            client_pick = target_c[0]
            client_pick.location = Location.DEVICE
            picks.append(client_pick)
        except:
            client_pick = None
        if self.use_split:
            try:
                split_pick = target_sp[0]
                split_pick.location = Location.SPLIT
                picks.append(split_pick)
            except:
                split_pick = None


        #if client_pick == None and server_pick == None and split_pick == None:
        #print(picks)
        if (not self.server_only) and len(picks) == 0:
            target_c = tApp.models 
            target_c.sort(key=lambda x:x.compute_energy)
            if target_c[0].compute_energy < self.energy_budget:
                client_pick = target_c[0]
                client_pick.location = Location.DEVICE
                picks.append(client_pick)

        if connected and len(picks) == 0:
            target_s = tApp.models 
            target_s.sort(key=lambda x:x.s_compute_latency)
            if target_s[0].s_compute_latency * server_cost < self.cost_budget:
                server_pick = target_s[0]
                server_pick.location = Location.SERVER
                picks.append(server_pick)

        if len(picks) == 0:
            tApp.pick = None
            tApp.status = Location.NOTRUNNING
            tApp.res.append((i, 0, tApp.status))
            return

        picks.sort(key=lambda x:x.accuracy, reverse=True)
        # TODO: tie break
        pick = picks[0]
        if pick.location == Location.SERVER:
            self.cost_budget -= server_pick.s_compute_latency * server_cost
            self.energy_budget -= send_energy

            if tApp.status != Location.SERVER:
                try:
                    self.in_cache[tApp.status].remove(tApp)
                except KeyError:
                    pass
                self.in_cache[pick.location].append(tApp)

        elif pick.location == Location.DEVICE:
            if tApp.status != pick.location:
                self.energy_budget -= client_pick.loading_energy
                tApp.last_swapin = i
                try:
                    self.in_cache[tApp.status].remove(tApp)
                except KeyError:
                    pass
                self.in_cache[pick.location].append(tApp)

            self.energy_budget -= client_pick.compute_energy

        elif pick.location == Location.SPLIT:
            if tApp.status != pick.location:
                self.energy_budget -= client_pick.sp_loading_energy
                tApp.last_swapin_split = i
            self.energy_budget -= client_pick.sp_compute_energy
            self.cost_budget -= server_pick.sp_s_compute_latency * server_cost
            self.energy_budget -= send_energy

        else:
            print("ERROR")
            exit()

        tApp.status = pick.location
        tApp.pick = pick

        """ 
        if server_pick != None and (client_pick == None or server_pick.accuracy > client_pick.accuracy):
            if tApp.status != Location.SERVER:
                #print(i, "client->server")
                moves += 1
                tApp.status = Location.SERVER  # server!!
            self.cost_budget -= server_pick.s_compute_latency * server_cost
            pick = server_pick
            self.energy_budget -= send_energy
        elif server_pick != None and (server_pick.accuracy == client_pick.accuracy and prev_acc == server_pick.accuracy):
            if tApp.status == Location.SERVER:
                self.cost_budget -= server_pick.s_compute_latency * server_cost
                pick = server_pick
                self.energy_budget -= send_energy
            else:
                tApp.status = Location.DEVICE 
                self.energy_budget -= client_pick.compute_energy
                pick = client_pick 
        else:
            if tApp.status != Location.DEVICE:
                #print(i, "server->client")
                moves += 1
                self.energy_budget -= client_pick.loading_energy
                tApp.last_swapin = i
            tApp.status = Location.DEVICE  # client
            self.energy_budget -= client_pick.compute_energy
            pick = client_pick 
        """

        tApp.res.append((i, pick.accuracy, tApp.status))
        #print(i, self.energy_budget, self.cost_budget)
        self.res.append((i, self.energy_budget, self.cost_budget))
        prev_acc = pick.accuracy


    def rununtil(self, trace, until=36000):
        moves = 0
        prev_acc = 0
        for cur in trace:
            i = cur[0]
            if i > until: break
            tApp = self.applications[cur[-1]][0]
            self.execute_task(i, until, tApp, cur)
            if cur[-1] == AppType.FACE and len(self.applications[AppType.FACE_V]) > 0:
                for tApp in self.applications[AppType.FACE_V]:
                    self.execute_task(i, until, tApp, cur, self.use_sharing)






# sheculder 2
"""
scheduler = Scheduler("test1", 5*3600, 0.0667)
scheduler.add_application(AppType.FACE, app1)
scheduler.add_application(AppType.SCENE, app2)
"""

# split
#scheduler.add_application(AppType.SCENE, app2)

import pickle
# connectivity test
"""
with open("disconnect.pcl", "rb") as f:
    conn = pickle.load(f)
    scheduler.set_connectivity(conn)
"""
#energy_budget = 2.*3600 # 5Wh
#cost_budget = 0.04 # dollar ($2/month)




import matplotlib.pyplot as plt 
#from mpltools import style
#style.use('ggplot')

def plot_energy_and_cost(ax, res, ylim):
    ln = ax.plot(map(lambda x:x[0], res), map(lambda x:x[1], res), c='b', label='Energy')
    ax.set_ylabel('Energy Budget (J)')
    ax.set_ylim(0,ylim)
    ax2 = ax.twinx()
    ln2 = ax2.plot(map(lambda x:x[0], res), map(lambda x:x[2], res), label='Cost')
    ax2.set_ylabel('Cost Budget ($)')
    ax.set_xlim(0,36000)
    lns = ln + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=3)


def depict_2(res, r1, r2, filename, ylim=18000):
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ln = plot_energy_and_cost(ax, res, ylim)
    ax = fig.add_subplot(312)
    ln3 = ax.plot(map(lambda x:x[0], r1), map(lambda x:x[1], r1), label='App1')
    ln4 = ax.plot(map(lambda x:x[0], r2), map(lambda x:x[1], r2), label='App1')
    ax.set_xlim(0,36000)
    ax.set_ylabel('Accuracy (%)')
    ax = fig.add_subplot(313)
    ln4 = ax.step(map(lambda x:x[0], r1), map(lambda x:x[2], r1), where='post', label='Acc')
    ln4 = ax.step(map(lambda x:x[0], r2), map(lambda x:x[2], r2), where='post', label='Acc')
    ax.set_xlim(0,36000)
    ax.set_ylim(0.8, 3.2)
    ax.set_yticks([3,2, 1])
    ax.set_yticklabels(['server','split', 'client'])
    plt.show()
    fig.savefig(filename, bbox_inches='tight')

def depict_m(res, rs, filename, ylim=18000):
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ln = plot_energy_and_cost(ax, res, ylim)
    ax = fig.add_subplot(312)
    for r in rs:
        ln3 = ax.plot(map(lambda x:x[0], r), map(lambda x:x[1], r), label='App1')
    ax.set_xlim(0,36000)
    ax.set_ylabel('Accuracy (%)')
    ax = fig.add_subplot(313)
    for r in rs:
        ln4 = ax.step(map(lambda x:x[0], r), map(lambda x:x[2], r), where='post', label='Acc')
    ax.set_xlim(0,36000)
    ax.set_ylim(0.8, 3.2)
    ax.set_yticks([3,2, 1])
    ax.set_yticklabels(['server','split', 'client'])
    plt.show()
    fig.savefig(filename, bbox_inches='tight')


def depict(res, r1, filename, ylim=18000):
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ln = plot_energy_and_cost(ax, res, ylim)
    ax = fig.add_subplot(312)
    ln3 = ax.plot(map(lambda x:x[0], r1), map(lambda x:x[1], r1), label='App1')
    ax.set_xlim(0,36000)
    ax.set_ylabel('Accuracy (%)')
    ax = fig.add_subplot(313)
    ln4 = ax.step(map(lambda x:x[0], r1), map(lambda x:x[2], r1), where='post', label='Acc')
    ax.set_xlim(0,36000)
    ax.set_ylim(0.8, 3.2)
    ax.set_yticks([3,2, 1])
    ax.set_yticklabels(['server','split', 'client'])
    plt.show()
    fig.savefig(filename, bbox_inches='tight')

def depict_special(res, r1, filename, sp_trace, ylim=18000):
    fig = plt.figure()
    ax = fig.add_subplot(411)
    ln = ax.plot(map(lambda x:x[0], res), map(lambda x:x[1], res), c='b', label='Energy')
    ax.set_ylabel('Energy Budget (J)')
    ax.set_ylim(0,ylim)
    ax2 = ax.twinx()
    ln2 = ax2.plot(map(lambda x:x[0], res), map(lambda x:x[2], res), label='Cost')
    ax2.set_ylabel('Cost Budget ($)')
    ax.set_xlim(0,36000)
    lns = ln + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=3)
    ax = fig.add_subplot(412)
    ln3 = ax.plot(map(lambda x:x[0], r1), map(lambda x:x[1], r1), label='App1')
    ax.set_xlim(0,36000)
    ax.set_ylabel('Accuracy (%)')
    ax = fig.add_subplot(413)
    ln4 = ax.step(map(lambda x:x[0], r1), map(lambda x:x[2], r1), where='post', label='Acc')
    ax.set_xlim(0,36000)
    ax.set_ylim(0.8, 3.2)
    ax.set_yticks([3, 1])
    ax.set_yticklabels(['server', 'client'])
    ax = fig.add_subplot(414)
    ax.scatter(map(lambda x:x[0], sp_trace), map(lambda x:x[1], sp_trace))
    ax.set_xlim(0,36000)
    ax.set_ylim(0,200)
    plt.show()
    fig.savefig(filename, bbox_inches='tight')

param = model_pb2.ApplicationModel()
with open("deepface.prototxt") as f:
    google.protobuf.text_format.Merge(f.read(), param)

# load face trace
with open("poi_1.pcl", "rb") as f:
    trace = pickle.load(f)

with open("poi_2.pcl", "rb") as f:
    trace_2 = pickle.load(f)

with open("poi_5.pcl", "rb") as f:
    trace_5 = pickle.load(f)

with open("poi_100.pcl", "rb") as f:
    trace_100 = pickle.load(f)

#trace_5 = map(lambda x:(x, AppType.SCENE), trace_5)
#trace = trace + trace_5
#trace.sort(key=lambda x:x[0])


def multipleApps():
    global trace, trace_5
    app1 = Application("deepface", .2, param.models)
    app2 = Application("object-alex", 1, param2.models)
    scheduler = Scheduler("multi", 2*3600, 0.0667)
    trace = map(lambda x:(x, AppType.SCENE), trace)
    trace_5 = map(lambda x:(x, AppType.FACE), trace_5)
    trace = trace + trace_5
    trace.sort(key=lambda x:x[0])
    scheduler.add_application(AppType.FACE, app1)
    scheduler.add_application(AppType.SCENE, app2)
    scheduler.rununtil(trace)
    res = scheduler.res
    depict_2(res, app1.res, app2.res, "multi.pdf")

def split():
    app1 = Application("deepface", 1., param.models)
    scheduler = Scheduler("split", 2*3600, 0.0667)
    scheduler.add_application(AppType.FACE, app1)
    scheduler.use_split = True
    trace = map(lambda x:(x, AppType.FACE), trace)

    scheduler.rununtil(trace)
    res = scheduler.res
    depict(res, app1.res, "split.pdf")

def special():
    param2 = model_pb2.ApplicationModel()
    with open("special_models.prototxt") as f:
        google.protobuf.text_format.Merge(f.read(), param2)
    app1 = Application("deepface", .2, param.models, param2.models)

    with open("special_trace.pcl", "rb") as f:
        special_trace = pickle.load(f)
        special_trace = map(lambda x:(x[0], x[1], AppType.FACE), special_trace)
    scheduler = Scheduler("special", 2*3600, 0.0667)
    scheduler.use_specialize = True
    scheduler.add_application(AppType.FACE, app1)
    scheduler.rununtil(special_trace)
    res = scheduler.res
    depict_special(res, app1.res, "special.pdf", special_trace, 2*3600)
   
def sharing():
    param = model_pb2.ApplicationModel()
    with open("sharing_test.prototxt") as f:
        google.protobuf.text_format.Merge(f.read(), param)

    app1 = Application("deepface", .2, param.models)
    scheduler = Scheduler("sharing", 2*3600, 0.0667)
    scheduler.add_application(AppType.FACE, app1)

    param_g = model_pb2.ApplicationModel()
    with open("sharing_gender.prototxt") as f:
        google.protobuf.text_format.Merge(f.read(), param_g)
    app2 = Application("deepface-g", .2, param.models)
    scheduler.add_application(AppType.FACE_V, app2)

    param_a = model_pb2.ApplicationModel()
    with open("sharing_age.prototxt") as f:
        google.protobuf.text_format.Merge(f.read(), param_g)
    app3 = Application("deepface-a", .2, param.models)
    scheduler.add_application(AppType.FACE_V, app3)

    param_r = model_pb2.ApplicationModel()
    with open("sharing_race.prototxt") as f:
        google.protobuf.text_format.Merge(f.read(), param_g)
    app4 = Application("deepface-r", .2, param.models)
    scheduler.add_application(AppType.FACE_V, app4)

    trace = map(lambda x:(x, AppType.FACE), trace_5)
    scheduler.rununtil(trace)

    res = scheduler.res
    depict_m(res, [app1.res, app2.res, app3.res, app4.res],"nosharing.pdf", 2*3600)

def sharing_t():
    param = model_pb2.ApplicationModel()
    with open("sharing_test.prototxt") as f:
        google.protobuf.text_format.Merge(f.read(), param)

    app1 = Application("deepface", .2, param.models)
    scheduler = Scheduler("sharing", 2*3600, 0.0667)
    scheduler.add_application(AppType.FACE, app1)

    param_g = model_pb2.ApplicationModel()
    with open("sharing_gender.prototxt") as f:
        google.protobuf.text_format.Merge(f.read(), param_g)
    app2 = Application("deepface-g", .2, param_g.models)
    scheduler.add_application(AppType.FACE_V, app2)

    param_a = model_pb2.ApplicationModel()
    with open("sharing_age.prototxt") as f:
        google.protobuf.text_format.Merge(f.read(), param_a)
    app3 = Application("deepface-a", .2, param_a.models)
    scheduler.add_application(AppType.FACE_V, app3)

    param_r = model_pb2.ApplicationModel()
    with open("sharing_race.prototxt") as f:
        google.protobuf.text_format.Merge(f.read(), param_r)
    app4 = Application("deepface-r", .2, param_r.models)
    scheduler.add_application(AppType.FACE_V, app4)

    trace = map(lambda x:(x, AppType.FACE), trace_5)
    scheduler.use_sharing = True
    scheduler.rununtil(trace)

    res = scheduler.res
    depict_m(res, [app1.res, app2.res, app3.res, app4.res],"sharing.pdf", 2*3600)


#special() 
#sharing()
#sharing_t()
#multipleApps()
def scenario(i, desc):
    global trace, trace_5, trace_100, trace_2
    scheduler = Scheduler("sharing", 0.5*3600, 0.0667)
    print(i)

    with open("scenarios/special_s%d.pcl" % i, "rb") as f:
        special_trace = pickle.load(f)
        special_trace = map(lambda x:(x[0], x[1], AppType.FACE), special_trace)
    with open("scenarios/connectivity_s%d.pcl" % i, "rb") as f:
        conn = pickle.load(f)
        scheduler.set_connectivity(conn)

    #load app for 1
    if i == 1:

        param_sp = model_pb2.ApplicationModel()
        with open("special_models.prototxt") as f:
            google.protobuf.text_format.Merge(f.read(), param_sp)

        param = model_pb2.ApplicationModel()
        with open("sharing_test.prototxt") as f:
            google.protobuf.text_format.Merge(f.read(), param)

        app1 = Application("deepface", .2, param.models, param_sp.models)
        scheduler.add_application(AppType.FACE, app1)

        param_g = model_pb2.ApplicationModel()
        with open("sharing_gender.prototxt") as f:
            google.protobuf.text_format.Merge(f.read(), param_g)
        app2 = Application("deepface-g", .2, param_g.models)
        scheduler.add_application(AppType.FACE_V, app2)

        param_a = model_pb2.ApplicationModel()
        with open("sharing_age.prototxt") as f:
            google.protobuf.text_format.Merge(f.read(), param_a)
        app3 = Application("deepface-a", .2, param_a.models)
        scheduler.add_application(AppType.FACE_V, app3)

        param_r = model_pb2.ApplicationModel()
        with open("sharing_race.prototxt") as f:
            google.protobuf.text_format.Merge(f.read(), param_r)
        app4 = Application("deepface-r", .2, param_r.models)

        param2 = model_pb2.ApplicationModel()
        with open("model_as.prototxt") as f:
            google.protobuf.text_format.Merge(f.read(), param2)

        app_o = Application("object", .5, param2.models)
        scheduler.add_application(AppType.OBJECT, app_o)


        param_s = model_pb2.ApplicationModel()
        with open("scene.prototxt") as f:
            google.protobuf.text_format.Merge(f.read(), param_s)
        scene = Application("scene", .01, param_s.models)
        scheduler.add_application(AppType.SCENE, scene)

        trace_2 = map(lambda x:(x, AppType.OBJECT), trace_2)

        trace_100 = map(lambda x:(x, AppType.SCENE), trace_100)
        trace_100 = trace_100 + trace_2 + special_trace 
        trace_100.sort(key=lambda x:x[0])
        scheduler.use_sharing = True
        #scheduler.use_specialize = False
        #scheduler.server_only = True
        scheduler.client_only = True
        scheduler.rununtil(trace_100)
        res = scheduler.res
        apps = [app1, app2, app3, app4, app_o, scene]

        all_acc = []
        for app in apps:
            all_acc.extend(map(lambda x:x[1], app.res))

        print("average acc:", sum(all_acc)/float(len(all_acc)))

    if i == 2:
        param_sp = model_pb2.ApplicationModel()
        with open("special_models.prototxt") as f:
            google.protobuf.text_format.Merge(f.read(), param_sp)

        param = model_pb2.ApplicationModel()
        with open("model_sample.prototxt") as f:
            google.protobuf.text_format.Merge(f.read(), param)

        app1 = Application("deepface", .2, param.models, param_sp.models)
        scheduler.add_application(AppType.FACE, app1)

        param2 = model_pb2.ApplicationModel()
        with open("model_as.prototxt") as f:
            google.protobuf.text_format.Merge(f.read(), param2)

        app2 = Application("object", .5, param2.models)
        scheduler.add_application(AppType.OBJECT, app2)

        apps = [app1, app2]
        trace_2 = map(lambda x:(x, AppType.OBJECT), trace_2)
        trace_2 = trace_2 + special_trace 
        trace_2.sort(key=lambda x:x[0])

        #scheduler.use_specialize = False
        #scheduler.server_only = True
        #scheduler.client_only = True
        scheduler.rununtil(trace_2)
        depict_m(scheduler.res, [app1.res, app2.res], "sc2.pdf", 0.5*3600)
        all_acc = []
        for app in apps:
            all_acc.extend(map(lambda x:x[1], app.res))

        print("average acc:", sum(all_acc)/float(len(all_acc)))

    if i == 3:
        print("running 3")
        param = model_pb2.ApplicationModel()
        with open("sharing_test.prototxt") as f:
            google.protobuf.text_format.Merge(f.read(), param)
        apps = []
        app1 = Application("deepface", .2, param.models)
        scheduler.add_application(AppType.FACE, app1)
        apps.append(app1)

        param_g = model_pb2.ApplicationModel()
        with open("sharing_gender.prototxt") as f:
            google.protobuf.text_format.Merge(f.read(), param_g)
        app2 = Application("deepface-g", .2, param_g.models)
        scheduler.add_application(AppType.FACE_V, app2)
        apps.append(app2)

        param_a = model_pb2.ApplicationModel()
        with open("sharing_age.prototxt") as f:
            google.protobuf.text_format.Merge(f.read(), param_a)
        app3 = Application("deepface-a", .2, param_a.models)
        scheduler.add_application(AppType.FACE_V, app3)
        apps.append(app3)

        param_r = model_pb2.ApplicationModel()
        with open("sharing_race.prototxt") as f:
            google.protobuf.text_format.Merge(f.read(), param_r)
        app4 = Application("deepface-r", .2, param_r.models)
        scheduler.add_application(AppType.FACE_V, app4)
        apps.append(app4)

        param_s = model_pb2.ApplicationModel()
        with open("scene.prototxt") as f:
            google.protobuf.text_format.Merge(f.read(), param_s)
        scene = Application("scene", .01, param_s.models)
        scheduler.add_application(AppType.SCENE, scene)

        trace_100 = map(lambda x:(x, AppType.SCENE), trace_100)
        trace_100 = trace_100 + special_trace 
        trace_100.sort(key=lambda x:x[0])
        apps.append(scene)

        #scheduler.use_sharing = True
        #scheduler.server_only = True
        scheduler.client_only = True
        scheduler.rununtil(trace_100)
        res = scheduler.res
        depict_m(res, [app1.res, app2.res, scene.res], "sc3.pdf", 0.5*3600)

        all_acc = []
        for app in apps:
            all_acc.extend(map(lambda x:x[1], app.res))

        print("average acc:", sum(all_acc)/float(len(all_acc)))


#scenario(3, "")
"""
ax = fig.add_subplot(311)
ln = ax.plot(map(lambda x:x[0], res), map(lambda x:x[1], res), c='b', label='Energy')
ax.set_ylabel('Energy Budget (J)')
ax.set_ylim(0,18000)
ax2 = ax.twinx()
ln2 = ax2.plot(map(lambda x:x[0], res), map(lambda x:x[2], res), label='Cost')
ax2.set_ylabel('Cost Budget ($)')
ax.set_xlim(0,36000)
lns = ln + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=3)
ax = fig.add_subplot(312)
r1 = app1.res
r2 = app2.res
ln3 = ax.plot(map(lambda x:x[0], r1), map(lambda x:x[1], r1), label='App1')
ln4 = ax.plot(map(lambda x:x[0], r2), map(lambda x:x[1], r2), label='App1')
ax.set_ylim(45,80)
ax.set_ylabel('Accuracy (%)')
ax.set_xlim(0,36000)
ax = fig.add_subplot(313)
ln4 = ax.step(map(lambda x:x[0], r1), map(lambda x:x[2], r1), where='post', label='Acc')
ln4 = ax.step(map(lambda x:x[0], r2), map(lambda x:x[2], r2), where='post', label='Acc')
ax.set_xlim(0,36000)
ax.set_ylim(0.8, 3.2)
ax.set_yticks([3, 1])
ax.set_yticklabels(['server', 'client'])
#plt.show()
ax = fig.add_subplot(411)
ln = ax.plot(map(lambda x:x[0], res), map(lambda x:x[1], res), c='b', label='Energy')
ax.set_ylabel('Energy Budget (J)')
ax.set_ylim(0,18000)
ax2 = ax.twinx()
ln2 = ax2.plot(map(lambda x:x[0], res), map(lambda x:x[2], res), label='Cost')
ax2.set_ylabel('Cost Budget ($)')
ax.set_xlim(0,36000)
lns = ln + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=3)
ax = fig.add_subplot(412)
r1 = app1.res
r2 = app2.res
ln3 = ax.plot(map(lambda x:x[0], r1), map(lambda x:x[1], r1), label='App1')
ln4 = ax.plot(map(lambda x:x[0], r2), map(lambda x:x[1], r2), label='App1')
ax.set_ylim(45,80)
ax.set_ylabel('Accuracy (%)')
ax.set_xlim(0,36000)
ax = fig.add_subplot(413)
ln4 = ax.step(map(lambda x:x[0], r1), map(lambda x:x[2], r1), where='post', label='Acc')
ln4 = ax.step(map(lambda x:x[0], r2), map(lambda x:x[2], r2), where='post', label='Acc')
ax.set_xlim(0,36000)
ax.set_ylim(0.8, 3.2)
ax.set_yticks([3, 1])
ax.set_yticklabels(['server', 'client'])
ax = fig.add_subplot(414)
ln4 = ax.step(map(lambda x:x[0], conn), map(lambda x:x[1], conn), where='post', label='Acc')
ax.set_ylabel('Connectivity')
ax.set_xlim(0,36000)
ax.set_ylim(-0.2,1.2)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Disconnected', 'Connected'])

#fig.savefig('schedule_2wh_004.pdf', bbox_inches='tight')
#fig.savefig('schedule2.pdf', bbox_inches='tight')
fig.savefig('schedule_disconn.pdf', bbox_inches='tight')
"""


