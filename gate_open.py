
def set_gate_open(angle=90, pin=3):
    """
    Opens the gate (servo). On non-Pi systems, prints a simulation message.
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
        print("[gate_open] Gate opened (servo).")
    except Exception as e:
        print(f"[gate_open] Simulated gate open (no GPIO): {e}")

if __name__ == "__main__":
    set_gate_open()
