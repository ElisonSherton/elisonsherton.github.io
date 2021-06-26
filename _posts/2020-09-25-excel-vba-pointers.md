---
layout: post
title: Understanding the Basics of Excel VBA Macros
published: true
categories: ['Miscellaneous']
---

This is an example post and will not have detailed explanations. It is intended to be a notebook for me to revisit concepts related to VBA from time to time.

- This is how we define a function in VBA

```VB
Function fname (a As Double, b As Double) as Double
    ***
    ***
    fname = result
End Function
```

- Borrowing an excel function is possible using the syntax
```
Application.WorksheetFunction.fname()
```

- Object Hierarchy in VBA

```json
{
    "Application":{
        "Addin":{},
        "Window":{},
        "WorksheetFunction":{},
        "Workbook":{
                    "Chart":{},
                    "Name":{},
                    "VBProject":{},
                    "Window":{},
                    "Worksheet":{
                                   "Comment":{},
                                   "Hyperlink":{},
                                   "PageSetup":{},
                                   "Range":{}
                                }

                   }
    }
}
```

- Copying one range to another

```
Range("A1:A4").Copy Range("C1:C4")
```

- Specifying when there is an error in the code, add an exit sub before here because otherwise error handling code will otherwise be run anyway.

```VB
On Error GoTo Here
code
Here:
code
```

- General Conditional in VBA

```VB
If condition Then
    code
Elseif condition2 Then
    code
Else
    code
End If
```

- Looping constructs in VBA

**For Next Loop**
```VB
Dim sum as Double
For i = 1 to N
    sum = sum + 1
Next i
```

**Do Loop**
```VB
Do
    code
    If condition Then Exit Do
    code
Loop
```


- To avoid user from seeing unnecessary screen updations
```VB
Application.ScreenUpdating = False
```


