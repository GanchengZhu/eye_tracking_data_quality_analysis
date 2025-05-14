# Experiment Source Code

## Experiment 1: Smartphone Eye Tracking Data Quality with EyeLink

You can found Android App source code [here](https://github.com/GanchengZhu/DataQualityWithEyeLink).

## Experiment 2: Smartphone Eye Tracking for Depression Symptom Detection

Android App source code [here](https://github.com/GanchengZhu/SmartphoneEyeTrackingDepression).

## Access to Smartphone Eye Tracking SDK

Please send an email to zhiguo@zju.edu.cn. Upon successful processing of your request,
you will receive an email containing the Smartphone Eye Tracking SDK.

### Email Prompt

Here’s a template for your request email. Please keep the subject line unchanged:

```
Subject: Request for Access to the Smartphone Eye Tracking SDK

Dear Prof. Zhiguo Wang,

I hope this message finds you well.

My name is [Your Name], and I am a [student/researcher] at [Your Affiliation]. I am writing to request the Smartphone Eye Tracking SDK.

I assure you that I will use this SDK solely for academic and research purposes and will not utilize it for commercial activities or share it with others.

Thank you for considering my request. I look forward to receiving access to the SDK.

Best regards,
[Your Name]
```

---

# How to Integrate the SDK into Your App

## 1. Create a New Android Project  
Create a new Android project in Android Studio or use the template project provided by this Android SDK.

![screenshots/img.png](screenshots/img.png)
**Android Studio New Project Setup Diagram**  

**Note:** The SDK is primarily written in Java, with a small portion in Kotlin. While Kotlin can seamlessly call Java code, this SDK has not been fully tested with Kotlin. Therefore, it is recommended to use Java for integration.  

---

## 2. Gradle Integration of Local SDK  

1. **Create a `lib-gaze-tracker` folder** in your project’s root directory. 

    ![screenshots/img_1.png](screenshots/img_1.png)

2. Place `lib-gaze-tracker-release.aar` in this folder and configure its `build.gradle`:  
   ```groovy
   configurations.maybeCreate("default")
   artifacts.add("default", file('lib-gaze-tracker-release.aar'))
   ```  
   
3. Update `settings.gradle` in the project root:  
   ```groovy
   include ':lib-gaze-tracker'
   ```  
   
4. Add dependencies in the app module’s `build.gradle`:  
   ```groovy
   dependencies {
       // Load lib-gaze-tracker
       implementation project(path: ':lib-gaze-tracker')
   }
   ```  
   
5. - **Supported Architectures**: `armeabi-v7a` and `arm64-v8a` only.  
    ```groovy
   android {
      defaultConfig {
         ndk {
            abiFilters 'arm64-v8a', 'armeabi-v7a'
         }
      }
   }
   ```
---

## 3. Initialize the SDK

- Import classes in your activity class.
    ```java
    import org.gaze.tracker.core.GazeTracker;
    import org.gaze.tracker.widget.AutoFitSurfaceView;
    ```  
- Init gaze tracker.
    ```java
    private void initGazeTracker() {
        GazeTracker.create(this, (gazeTracker, initializationErrors) -> {
            Log.i(TAG, "Initialization callback");
            tracker = gazeTracker;
            tracker.setSessionName(getUserId());
            tracker.setErrorBarVisible(getSwitchState());
            tracker.drawCalibrationUI(() -> {
                Intent intent = new Intent(getBaseContext(), ExampleActivity.class);
                intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                startActivity(intent);
            });
        });
    }
  ```
- Call `initGazeTracker` at `onCreate` method.
    ```java
    @Override
    protected void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);
            setContentView(R.layout.activity_main);
    
            rootLayout = findViewById(R.id.root_container);
            surfaceView = findViewById(R.id.surface_view);
            surfaceView.getHolder().setFormat(PixelFormat.RGBA_8888);
            surfaceView.getHolder().addCallback(this);
    
            // check the camera permission
            // if the camera permission denied, request the camera  permission.
            if (ActivityCompat.checkSelfPermission(this, PERMISSIONS[0])
                    == PackageManager.PERMISSION_GRANTED) {
                initGazeTracker();
            } else {
                ActivityCompat.requestPermissions(this, PERMISSIONS, REQ_PERMISSION);
            }
    }
    ```

- The SDK need to previewer user's face via Android `SurfaceView`.
    ```Java
   @Override
    public void surfaceCreated(@NonNull SurfaceHolder surfaceHolder) {
        Log.d(TAG, "surfaceCreated");
        WindowManager windowManager = (WindowManager) this.getSystemService(Context.WINDOW_SERVICE);
        int degree = 0;
        if (windowManager != null) {
            Display display = windowManager.getDefaultDisplay();
            switch (display.getRotation()) {
                case Surface.ROTATION_0:
                    degree = 0; // portrait
                    break;
                case Surface.ROTATION_90:
                    degree = 90; // landscape right
                    break;
                case Surface.ROTATION_180:
                    degree = 180; // portrait upsize
                    break;
                case Surface.ROTATION_270:
                    degree = 270; // landscape left
                    break;
            }
        }
        if (degree == 0 || degree == 180)
            surfaceView.setAspectRatio(480, 640);
        else {
            surfaceView.setAspectRatio(640, 480);
        }

        surfaceView.post(() -> {
                    if (tracker != null) {
                        tracker.setPreviewSurface(surfaceView.getHolder().getSurface());
                    }
                }
        );
    }
  ```
## 4. Sample Gaze Data

- Your experiment android activity or fragment must be implemented `GazeCallback`, like
```Java
import org.gaze.tracker.bean.GazeSample;
import org.gaze.tracker.core.GazeTracker;
import org.gaze.tracker.enumeration.TrackingState;
import org.gaze.tracker.listener.GazeCallback;

public class ExampleActivity extends AppCompatActivity
        implements GazeCallback {
        
    @Override public void onGaze(GazeSample gazeSample) {
        if (gazeSample.getTrackingState() == TrackingState.SUCCESS) {
            isPointShow = true;
            x = gazeSample.getFilteredX();
            y = gazeSample.getFilteredY();
//            Log.i(TAG, "Calibrated: " + gazeSample.isHasCalibrated());
//            Log.i(TAG, String.format("x: %.2f, y: %.2f", x, y));
        } else {
            isPointShow = false;
        }

        dataBuffer.setLength(0);
        dataBuffer.append(gazeSample.getTimestamp()).append(",")
                .append(gazeSample.getTrackingState().getValue()).append(",")
                .append(gazeSample.isHasCalibrated() ? 1 : 0).append(",")
                .append(gazeSample.getRawX()).append(",").append(gazeSample.getRawY()).append(",")
                .append(gazeSample.getCalibratedX()).append(",").append(gazeSample.getCalibratedY())
                .append(",").append(gazeSample.getFilteredX()).append(",")
                .append(gazeSample.getFilteredY()).append(",").append(gazeSample.getLeftDistance())
                .append(",").append(gazeSample.getRightDistance()).append("\n");
        try {
            outputStreamWriter.write(dataBuffer.toString());
            outputStreamWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
```

- Get instance, add callback, and start sampling at `onCreate` method.
```Java
@Override protected void onCreate(@Nullable Bundle savedInstanceState) {
    gazeTracker = GazeTracker.getInstance();
    gazeTracker.addCallbacks(this);
    gazeTracker.startSampling();
}
```

---

# IOS SDK Documentation

IOS SDK iS developing. 
Now, we have a App Demo on the App Store, you can visit it [here](https://apps.apple.com/cn/app/tcci-mobile-et/id6723893485).

---

# Experiment Data Analysis

## Data Quality

Run the following code

```bash
cd smartphone_and_eyelink
python data_quality.py
```

## Plotting EyeLink and Phone Data

```bash
cd smartphone_and_eyelink
python participant_data_plotting.py right
```

## Histgram Plotting (Fig. 4)

```bash
cd smartphone_and_eyelink
python acc_pre_me_histgram_plotting.py
```

## Statistical Tests

```bash
cd smartphone_and_eyelink
python statistical_test.py
```

---
