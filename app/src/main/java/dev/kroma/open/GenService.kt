package dev.kroma.open

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager

/**
 * Keeps a generation alive with the screen off: a foreground service (the ongoing progress
 * notification) plus a partial wake lock. The work itself still runs on MainActivity's threads;
 * this only stops Android from freezing the process while they run. A picture takes several minutes
 * and the model download much longer: the phone going to sleep mid-run is the normal case.
 */
class GenService : Service() {
    private var wake: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Lang.load(this)      // the service can be the first thing alive in a restarted process
        val n = progress(this, intent?.getStringExtra("text") ?: tr(R.string.preparing), 0, 0)
        if (Build.VERSION.SDK_INT >= 34) startForeground(ID_RUN, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        else startForeground(ID_RUN, n)
        if (wake == null) wake = (getSystemService(Context.POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "kroma:gen").apply { acquire(3 * 60 * 60 * 1000L) }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        wake?.let { if (it.isHeld) it.release() }; wake = null
        super.onDestroy()
    }

    companion object {
        private const val CH_RUN = "gen_run"; private const val CH_DONE = "gen_done"
        private const val ID_RUN = 1; private const val ID_DONE = 2

        private fun channels(ctx: Context) {
            val nm = ctx.getSystemService(NotificationManager::class.java)
            nm.createNotificationChannel(NotificationChannel(CH_RUN, tr(R.string.progress), NotificationManager.IMPORTANCE_LOW))
            nm.createNotificationChannel(NotificationChannel(CH_DONE, tr(R.string.done), NotificationManager.IMPORTANCE_DEFAULT))
        }

        private fun openApp(ctx: Context) = PendingIntent.getActivity(ctx, 0,
            Intent(ctx, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)

        private fun progress(ctx: Context, text: String, cur: Int, max: Int): Notification {
            channels(ctx)
            return Notification.Builder(ctx, CH_RUN).setSmallIcon(android.R.drawable.ic_menu_gallery)
                .setContentTitle(tr(R.string.kroma_is_working)).setContentText(text).setOngoing(true).setOnlyAlertOnce(true)
                .setProgress(max, cur, max == 0).setContentIntent(openApp(ctx)).build()
        }

        fun start(ctx: Context) {
            runCatching { ctx.startForegroundService(Intent(ctx, GenService::class.java)) }
        }

        fun update(ctx: Context, text: String, cur: Int, max: Int) {
            runCatching { ctx.getSystemService(NotificationManager::class.java).notify(ID_RUN, progress(ctx, text, cur, max)) }
        }

        /** Ends the service; [doneText] non-null posts a separate "done" notification that stays. */
        fun stop(ctx: Context, doneText: String?) {
            runCatching { ctx.stopService(Intent(ctx, GenService::class.java)) }
            if (doneText == null) return
            runCatching {
                channels(ctx)
                ctx.getSystemService(NotificationManager::class.java).notify(ID_DONE,
                    Notification.Builder(ctx, CH_DONE).setSmallIcon(android.R.drawable.ic_menu_gallery)
                        .setContentTitle(tr(R.string.done)).setContentText(doneText).setAutoCancel(true)
                        .setContentIntent(openApp(ctx)).build())
            }
        }
    }
}
