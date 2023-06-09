def getContactPoints(FootTrajSlice):
# FootTrajSlice = FeetTrajs['LF_shank_fixed_LF_FOOT']
    ContactShow  = []
    ContactPoints = []
    BIGWINDOW = 500
    SMALLWINDOW = 40
    SMALLWINDOWRANGE = 0.015
    # Calculate the mean of the big window
    for bigWindowCount in range(len(FootTrajSlice)//BIGWINDOW+1):
        bWFront = bigWindowCount*BIGWINDOW
        bWBack = bWFront+BIGWINDOW
        if bWBack < len(FootTrajSlice) -1:
            bWBack = bWBack
            ref = (FootTrajSlice[bWFront:bWBack, 2]).mean()
        else:
            # Do not update ref since the ref is not accurate
            bWBack = len(FootTrajSlice) -1

        # Calculate the range of the small window
        for smallWindowCount in range(bWBack - bWFront+1):
            sWFront = bWFront + smallWindowCount
            sWBack = sWFront + SMALLWINDOW
            if sWBack < len(FootTrajSlice) -1:
                sWBack = sWBack     
            else:
                # abondan last small window, since its range is not accurate
                break
            smallWindowData = FootTrajSlice[sWFront:sWBack, 2]
            # Find Contacts
            if abs(smallWindowData.max() - smallWindowData.min()) < SMALLWINDOWRANGE and smallWindowData.min()<ref:
                # ContactShow is used for visualization and filled by None when no contacts
                ContactShow.append(FootTrajSlice[sWFront, 2])
                ContactPoints.append(FootTrajSlice[sWFront])
            else:
                ContactShow.append(None)
    
    return ContactPoints, ContactShow