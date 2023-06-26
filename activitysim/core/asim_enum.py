from enum import IntEnum

# enum class for the different time periods
class TimePeriod(IntEnum):
    AM = 1
    PM = 2
    MD = 3
    EV = 4
    NT = 5

# enum class for the different tour purposes
class TourPurpose(IntEnum):
    work = 1
    school = 2
    univ = 3
    escort = 4
    shopping = 5
    othmaint = 6
    eatout = 7
    social = 8
    othdiscr = 9

    @property
    def __name2valuedict__(self):
        return {i.name: i.value for i in self}

# enum class for the different tour category
class TourCategory(IntEnum):
    mandatory = 1
    joint = 2
    non_mandatory = 3
    atwork = 4

    @property
    def __name2valuedict__(self):
        return {i.name: i.value for i in self}

# enum class for the different daily activity pattern
class CdapCategory(IntEnum):
    M = 1
    N = 2
    H = 3

# enum class for mandatory tour frequency
class MandatoryTourFrequency(IntEnum):
    na = 0
    work1 = 1
    work2 = 2
    school1 = 3
    school2 = 4
    work_and_school = 5

    @property
    def __name2valuedict__(self):
        return {i.name: i.value for i in self}

# enum class for joint tour frequency
class JointTourFrequency(IntEnum):
    na = 0
    j0_tours = 1
    j1_Shop = 2
    j1_Main = 3
    j1_Eat = 4
    j1_Visit = 5
    j1_Disc = 6
    j2_SS = 7
    j2_SM = 8
    j2_SE = 9
    j2_SV = 10
    j2_SD = 11
    j2_MM = 12
    j2_ME = 13
    j2_MV = 14
    j2_MD = 15
    j2_EE = 16
    j2_EV = 17
    j2_ED = 18
    j2_VV = 19
    j2_VD = 20
    j2_DD = 21


# enum class for tour composition
class TourComposition(IntEnum):
    na = 0
    adults = 1
    children = 2
    mixed = 3