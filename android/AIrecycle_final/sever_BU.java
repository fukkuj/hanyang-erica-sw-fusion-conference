package com.example.airecycle;

import android.annotation.TargetApi;
import android.app.Activity;
import android.content.DialogInterface;
import android.os.Build;
import android.os.Bundle;
import android.os.StrictMode;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.lang.annotation.Target;
import java.net.Socket;

public class ClientTest extends Activity {
    private Socket socket;
    BufferedReader socket_in;
    PrintWriter socket_out;
    EditText input;
    Button button;
    TextView output;
    String data;

    @TargetApi(Build.VERSION_CODES.GINGERBREAD)
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        StrictMode.enableDefaults();
        try {
            socket = new Socket("localhost", 5558);
            socket_out = new PrintWriter(socket.getOutputStream(), true);
            socket_in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        } catch (IOException e) {
            e.printStackTrace();
        }
        input = (EditText) findViewById(R.id.input);
        button = (Button) findViewById(R.id.button);
        output = (TextView) findViewById(R.id.output);
        button.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                String data = input.getText().toString();
                if( data != null) {
                    Log.w("NETWORK" , " " + data);
                    socket_out.println(data);
                }
            }
        });

        Thread worker = new Thread() {
            public void run() {
                try {
                    while(true) {
                        data = socket_in.readLine();
                        output.post(new Runnable() {
                            @Override
                            public void run() {
                                output.setText(data);
                            }
                        });
                    }
                }catch (Exception e) {

                }
            }
        };
        worker.start();
    }
    @Override
    protected void onStop() {
        super.onStop();
        try{
            socket.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}






curl -H "Content-Type: application/json" \
     -H "Authorization: key=AAAAIO9fqLo:APA91bFk4VpDva55PgVLig5GtA8lBjPQHa_ynNfj9UjB8i8ZLb8lTgGsAIQiPYFiHm1M9Jc4vzmO2W7PtcrjkknH61QYuAkHjgJ-v-1FFJk_sMjrIXKaz3xcCNB-I1Cw4Pzw2eAtKOhD" \
     -d '{
           "notification": {
             "title": "New chat message!",
             "body": "There is a new message in FriendlyChat",
             "icon": "/images/profile_placeholder.png",
             "click_action": "http://localhost:5000"
           },
           "to": "fqvB5BwyY8g:APA91bGyyxjXqgeA8pKzLC8ydztLJlpHNg4-wNgxUwB9KfUWadBxtRHPmQDEo95rA1SQKcA5dGmb5GJkjD9eYnRAgh9l_JkcihhV6GiuTfvji5-SBwbLeB5aCtPBXASWSaNKun4QSo8R"
         }' \
     https://fcm.googleapis.com/fcm/send