
def set_gate_close(angle=180, pin=3):
    """
    Closes the gate (servo). On non-Pi systems, prints a simulation message.
    """
    try:
        import RPi.GPIO as GPIO  # type: ignore
        from time import sleep

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pin, GPIO.OUT)
        pwm = GPIO.PWM(pin, 50)
        pwm.start(0)

        duty = angle / 18 + 2
        GPIO.output(pin, True)
        pwm.ChangeDutyCycle(duty)
        sleep(1)
        GPIO.output(pin, False)
        pwm.ChangeDutyCycle(0)
        pwm.stop()
        GPIO.cleanup()
        print("[gate_close] Gate closed (servo).")
    except Exception as e:
        print(f"[gate_close] Simulated gate close (no GPIO): {e}")

if __name__ == "__main__":
    set_gate_close()
