import sys
import math
import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return f"{self.x},{self.y}"


class Vector:
    def __init__(self, pa: Point, pb: Point):
        # diferença em float, não converte para int (evita vetor nulo)
        self.x = float(pb.x) - float(pa.x)
        self.y = float(pb.y) - float(pa.y)

    def __str__(self):
        return f"{self.x},{self.y}"

    def norm(self):
        return math.hypot(self.x, self.y)


class Angle:
    def __init__(self, va: Vector, vb: Vector):
        self.va = va
        self.vb = vb

    def theta(self):
        norm_a = self.va.norm()
        norm_b = self.vb.norm()

        # evita divisão por zero (vetor nulo)
        if norm_a == 0 or norm_b == 0:
            return 0.0  # ou float("nan") se quiser marcar inválido

        dot = self.va.x * self.vb.x + self.va.y * self.vb.y
        cos_theta = dot / (norm_a * norm_b)

        # segurança contra erros numéricos (cos em [-1,1])
        cos_theta = max(-1.0, min(1.0, cos_theta))

        return math.degrees(math.acos(cos_theta))


class Distance:
    def __init__(self, pa: Point, pb: Point):
        dx = float(pb.x) - float(pa.x)
        dy = float(pb.y) - float(pa.y)
        self.value = math.hypot(dx, dy)

    def dist(self):
        return self.value


def checkArg():
    if len(sys.argv) != 2:
        print("please give me file")
        sys.exit(0)


def readFile(filename):
    points = []
    f = open(filename, "r")
    for line in f.readlines():
        line = line.strip(" \t\n\r")
        x = line.split(",")[0]
        y = line.split(",")[1]
        points.append(Point(x, y))
    f.close()
    return points


def getCross(va, vb):
    return va.x * vb.y - va.y * vb.x


def getODI(pa, pb, pc, pd, pe, pf, pg, ph):
    va = Vector(pa, pb)
    vb = Vector(pc, pd)
    vc = Vector(pe, pf)
    vd = Vector(pg, ph)

    aa = Angle(va, vb).theta()
    ab = Angle(vc, vd).theta()
    cb = getCross(vc, vd)

    if cb < 0:
        ab = -ab

    return aa + ab


def getAPDI(pa, pb, pc, pd, pe, pf, pg, ph, pi, pj):
    va = Vector(pa, pb)
    vb = Vector(pc, pd)
    vc = Vector(pe, pf)
    vd = Vector(pg, ph)
    ve = Vector(pi, pj)

    aa = Angle(va, vb).theta()
    ab = Angle(vb, vc).theta()
    ac = Angle(vd, ve).theta()

    cb = getCross(vb, vc)
    cc = getCross(vd, ve)

    if cb > 0:
        ab = -ab
    if cc < 0:
        ac = -ac

    return aa + ab + ac


def writeFile(
    filename,
    points,
    ANBtype,
    SNBtype,
    SNAtype,
    ODItype,
    APDItype,
    FHItype,
    FMAtype,
    mwtype,
):
    f = open(filename, "w")
    for point in points:
        f.write(str(point) + "\n")
    f.write(ANBtype + "\n")
    f.write(SNBtype + "\n")
    f.write(SNAtype + "\n")
    f.write(ODItype + "\n")
    f.write(APDItype + "\n")
    f.write(FHItype + "\n")
    f.write(FMAtype + "\n")
    f.write(mwtype + "\n")
    f.close()


def classification(points):
    results = {}

    # --- ANB ---
    va = Vector(points[1], points[0])
    vb = Vector(points[1], points[5])
    vc = Vector(points[1], points[0])
    vd = Vector(points[1], points[4])

    ANB = Angle(vc, vd).theta() - Angle(va, vb).theta()
    if ANB < 3.2:
        ANBtype = "3"
    elif ANB > 5.7:
        ANBtype = "2"
    else:
        ANBtype = "1"
    results["ANB"] = {"value": ANB, "class": ANBtype}

    # --- SNB ---
    va = Vector(points[1], points[0])
    vb = Vector(points[1], points[5])
    SNB = Angle(va, vb).theta()
    if SNB < 74.6:
        SNBtype = "2"
    elif SNB > 78.7:
        SNBtype = "3"
    else:
        SNBtype = "1"
    results["SNB"] = {"value": SNB, "class": SNBtype}

    # --- SNA ---
    va = Vector(points[1], points[0])
    vb = Vector(points[1], points[4])
    SNA = Angle(va, vb).theta()
    if SNA < 79.4:
        SNAtype = "3"
    elif SNA > 83.2:
        SNAtype = "2"
    else:
        SNAtype = "1"
    results["SNA"] = {"value": SNA, "class": SNAtype}

    # --- ODI ---
    ODI = getODI(
        points[7],
        points[9],
        points[5],
        points[4],
        points[3],
        points[2],
        points[16],
        points[17],
    )
    if ODI < 68.4:
        ODItype = "3"
    elif ODI > 80.5:
        ODItype = "2"
    else:
        ODItype = "1"
    results["ODI"] = {"value": ODI, "class": ODItype}

    # --- APDI ---
    APDI = getAPDI(
        points[2],
        points[3],
        points[1],
        points[6],
        points[4],
        points[5],
        points[3],
        points[2],
        points[16],
        points[17],
    )
    if APDI < 77.6:
        APDItype = "2"
    elif APDI > 85.2:
        APDItype = "3"
    else:
        APDItype = "1"
    results["APDI"] = {"value": APDI, "class": APDItype}

    # --- FHI ---
    pfh = Distance(points[0], points[9]).dist()
    afh = Distance(points[1], points[7]).dist()
    ratio = pfh / afh if afh != 0 else 0
    if ratio < 0.65:
        FHItype = "3"
    elif ratio > 0.75:
        FHItype = "2"
    else:
        FHItype = "1"
    results["FHI"] = {"value": ratio, "class": FHItype}

    # --- FMA ---
    va = Vector(points[0], points[1])
    vb = Vector(points[9], points[8])
    FMA = Angle(va, vb).theta()
    if FMA < 26.8:
        FMAtype = "3"
    elif FMA > 31.4:
        FMAtype = "2"
    else:
        FMAtype = "1"
    results["FMA"] = {"value": FMA, "class": FMAtype}

    # --- MW ---
    mw = Distance(points[10], points[11]).dist() / 10
    if points[11].x < points[10].x:
        mw = -mw
    if mw >= 2:
        if mw <= 4.5:
            mwtype = "1"
        else:
            mwtype = "4"
    elif mw == 0:
        mwtype = "2"
    else:
        mwtype = "3"
    results["MW"] = {"value": mw, "class": mwtype}

    return results
