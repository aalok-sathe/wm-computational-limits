# wm-computational-limits

Are there computational limits on human working memory (WM) capacity aside from anatomical limits?

The directory `workingmem.task.SIR` contains a version of the Store-Ignore-Recall (SIR) task used in human
experiments to tax working memory (CITE). The task involves storing and recalling items stored in
virtual WM 'slots', here, 'registers'. In humans, the task requires active role-addressable 
maintenance of information. In computational models, the retrieval may be facilitated by a number
of strategies, 

Examples look of this sort:
```
St reg_54 item_4 diff St reg_54 item_81 diff St reg_60 item_81 diff St reg_60 item_81 diff St reg_54 item_81 same St reg_54 item_81 same St reg_60 item_40 diff Ig reg_54 item_81 same St reg_54 item_81 same Ig reg_54 item_81 diff St reg_60 item_81 diff St reg_54 item_81 diff St reg_60 item_40 diff St reg_60 item_4 diff Ig reg_54 item_81 same
```